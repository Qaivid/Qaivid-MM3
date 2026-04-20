"""Stripe billing routes: /pricing, /billing/checkout, /billing/success,
/billing/cancel, /billing/webhook.

All Stripe operations are guarded with graceful fallbacks when STRIPE_SECRET_KEY
is not configured — the pricing page still renders but checkout is disabled.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

from auth import (
    current_user,
    get_user_by_stripe_customer,
    login_required,
    update_user_plan,
)

logger = logging.getLogger(__name__)

billing_bp = Blueprint("billing", __name__)

FREE_PROJECT_LIMIT = 3

PLANS = {
    "free": {
        "name": "Free",
        "price_monthly": 0,
        "price_label": "$0 / mo",
        "features": [
            "3 projects",
            "Full storyboard pipeline",
            "AI-generated stills",
            "All languages",
        ],
        "limits": ["No commercial licence"],
        "stripe_price_env": None,
    },
    "pro": {
        "name": "Pro",
        "price_monthly": 29,
        "price_label": "$29 / mo",
        "features": [
            "Unlimited projects",
            "Priority AI generation",
            "Face-locked character stills",
            "Commercial licence",
            "All languages & custom styles",
        ],
        "limits": [],
        "stripe_price_env": "STRIPE_PRO_PRICE_ID",
    },
    "studio": {
        "name": "Studio",
        "price_monthly": 99,
        "price_label": "$99 / mo",
        "features": [
            "Everything in Pro",
            "Team seats",
            "Custom style packs",
            "API access",
            "Dedicated support",
        ],
        "limits": [],
        "stripe_price_env": "STRIPE_STUDIO_PRICE_ID",
    },
}


def _stripe():
    import stripe as _s
    key = os.getenv("STRIPE_SECRET_KEY")
    if not key:
        return None
    _s.api_key = key
    return _s


def _stripe_enabled() -> bool:
    return bool(os.getenv("STRIPE_SECRET_KEY"))


# ---------------------------------------------------------------------------
# Pricing page
# ---------------------------------------------------------------------------

@billing_bp.route("/pricing")
def pricing():
    user = current_user()
    current_plan = (user or {}).get("plan", "free")
    plan_expires_at = (user or {}).get("plan_expires_at")
    stripe_enabled = _stripe_enabled()
    return render_template(
        "pricing.html",
        plans=PLANS,
        current_plan=current_plan,
        plan_expires_at=plan_expires_at,
        stripe_enabled=stripe_enabled,
    )


# ---------------------------------------------------------------------------
# Checkout
# ---------------------------------------------------------------------------

@billing_bp.route("/billing/checkout", methods=["POST"])
@login_required
def checkout():
    if not _stripe_enabled():
        flash("Payment processing is not yet configured. Check back soon.", "info")
        return redirect(url_for("billing.pricing"))

    plan_key = request.form.get("plan", "").lower()
    if plan_key not in ("pro", "studio"):
        flash("Invalid plan selection.", "error")
        return redirect(url_for("billing.pricing"))

    price_env = PLANS[plan_key]["stripe_price_env"]
    price_id = os.getenv(price_env) if price_env else None
    if not price_id:
        flash("This plan is not yet available for purchase. Check back soon.", "info")
        return redirect(url_for("billing.pricing"))

    stripe = _stripe()
    user = current_user()

    try:
        base_url = request.host_url.rstrip("/")
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            customer_email=user["email"] if not user.get("stripe_customer_id") else None,
            customer=user.get("stripe_customer_id") or None,
            success_url=base_url + url_for("billing.success") + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=base_url + url_for("billing.cancel"),
            metadata={"user_id": str(user["id"]), "plan": plan_key},
            subscription_data={"metadata": {"user_id": str(user["id"]), "plan": plan_key}},
        )
        return redirect(session.url, code=303)
    except Exception as exc:
        logger.exception("Stripe checkout error: %s", exc)
        flash("Could not start checkout. Please try again.", "error")
        return redirect(url_for("billing.pricing"))


# ---------------------------------------------------------------------------
# Success / Cancel return pages
# ---------------------------------------------------------------------------

@billing_bp.route("/billing/success")
@login_required
def success():
    return render_template("billing_success.html")


@billing_bp.route("/billing/cancel")
@login_required
def cancel():
    flash("Payment cancelled — you're still on the Free plan.", "info")
    return redirect(url_for("billing.pricing"))


# ---------------------------------------------------------------------------
# Stripe webhook
# ---------------------------------------------------------------------------

@billing_bp.route("/billing/webhook", methods=["POST"])
def webhook():
    stripe = _stripe()
    if not stripe:
        return "Stripe not configured", 503

    payload = request.get_data()
    sig = request.headers.get("Stripe-Signature", "")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    if not webhook_secret:
        logger.warning("Stripe webhook received but STRIPE_WEBHOOK_SECRET is not set — rejected")
        return "Webhook secret not configured", 503

    try:
        event = stripe.Webhook.construct_event(payload, sig, webhook_secret)
    except stripe.error.SignatureVerificationError:
        logger.warning("Stripe webhook signature verification failed")
        return "Bad signature", 400
    except Exception as exc:
        logger.error("Stripe webhook parse error: %s", exc)
        return "Bad payload", 400

    etype = event.get("type") or (event.get("object") or {}).get("type", "")
    obj = event.get("data", {}).get("object", {})

    if etype == "checkout.session.completed":
        _handle_checkout_completed(obj)
    elif etype in ("customer.subscription.deleted", "invoice.payment_failed"):
        customer_id = obj.get("customer")
        if customer_id:
            _downgrade_customer(customer_id)
    elif etype == "customer.subscription.updated":
        _handle_subscription_updated(obj)

    return "", 200


def _handle_checkout_completed(session_obj: dict) -> None:
    user_id = None
    plan_key = None

    meta = session_obj.get("metadata") or {}
    user_id = meta.get("user_id")
    plan_key = meta.get("plan")
    customer_id = session_obj.get("customer")
    subscription_id = session_obj.get("subscription")

    if not user_id or not plan_key:
        logger.error("checkout.session.completed missing metadata: %s", session_obj)
        return

    expires_at = None
    if subscription_id:
        try:
            stripe = _stripe()
            sub = stripe.Subscription.retrieve(subscription_id)
            ts = sub.get("current_period_end")
            if ts:
                expires_at = datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception as exc:
            logger.warning("Could not retrieve subscription %s: %s", subscription_id, exc)

    try:
        update_user_plan(
            user_id=int(user_id),
            plan=plan_key,
            stripe_customer_id=customer_id,
            plan_expires_at=expires_at,
        )
        logger.info("User %s upgraded to %s via checkout", user_id, plan_key)
    except Exception as exc:
        logger.exception("Failed to update plan for user %s: %s", user_id, exc)


def _handle_subscription_updated(sub_obj: dict) -> None:
    customer_id = sub_obj.get("customer")
    if not customer_id:
        return
    status = sub_obj.get("status")
    ts = sub_obj.get("current_period_end")
    expires_at = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
    user = get_user_by_stripe_customer(customer_id)
    if not user:
        return
    if status in ("active", "trialing"):
        update_user_plan(user["id"], user.get("plan", "pro"), plan_expires_at=expires_at)
    elif status in ("canceled", "unpaid", "past_due"):
        _downgrade_customer(customer_id)


def _downgrade_customer(customer_id: str) -> None:
    user = get_user_by_stripe_customer(customer_id)
    if not user:
        logger.warning("Webhook: no user found for customer %s", customer_id)
        return
    update_user_plan(user["id"], "free", plan_expires_at=None)
    logger.info("Downgraded user %s to free (customer %s)", user["id"], customer_id)
