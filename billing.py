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

# Credit costs (mirrors Qaivid 1 costPlanner.ts)
CREDIT_COSTS = {
    "still_image": 10,          # 1 AI still image = 10 credits (~10¢)
    "animatic_per_min": 100,    # 1 min animatic video = 100 credits
    "ai_video_per_min": 500,    # 1 min full AI video = 500 credits
    "creative_brief": 5,        # brief generation = 5 credits
    "transcription": 2,         # audio transcription = 2 credits
}

PLANS = {
    "free": {
        "name": "Free",
        "price_monthly": 0,
        "price_yearly": 0,
        "price_label": "$0",
        "credits_monthly": 0,
        "description": "Explore the pipeline with your first projects.",
        "features": [
            "3 projects",
            "Full storyboard pipeline",
            "AI-generated stills",
            "All 40+ languages",
        ],
        "limits": ["No commercial licence"],
        "stripe_price_env": None,
        "stripe_price_yearly_env": None,
        "popular": False,
        "yearly_saving": None,
    },
    "starter": {
        "name": "Starter",
        "price_monthly": 14.99,
        "price_yearly": 161.99,
        "price_yearly_monthly": 13.50,
        "price_label": "$14.99",
        "credits_monthly": 1500,
        "description": "For creators ready to produce their first AI video.",
        "features": [
            "1,500 credits / month",
            "~15 min of animatic video/month",
            "~3 min of full AI video/month",
            "AI Director's Brief & creative vision",
            "Character reference image generation",
            "Shot-by-shot storyboard with timing",
            "Final film stitching with audio",
        ],
        "limits": [],
        "stripe_price_env": "STRIPE_PRO_PRICE_ID",
        "stripe_price_yearly_env": None,
        "popular": False,
        "yearly_saving": "10%",
    },
    "pro": {
        "name": "Pro",
        "price_monthly": 39.99,
        "price_yearly": 407.99,
        "price_yearly_monthly": 34.00,
        "price_label": "$39.99",
        "credits_monthly": 4000,
        "description": "For serious creators producing at volume.",
        "features": [
            "4,000 credits / month",
            "~40 min of animatic video/month",
            "~8 min of full AI video/month",
            "Everything in Starter",
            "Subtitle designer & animated captions",
            "Cinematic colour grading & logos",
            "Post-production export suite",
        ],
        "limits": [],
        "stripe_price_env": "STRIPE_PRO_PRICE_ID",
        "stripe_price_yearly_env": None,
        "popular": True,
        "yearly_saving": "15%",
    },
    "studio": {
        "name": "Studio",
        "price_monthly": 199.99,
        "price_yearly": 1919.99,
        "price_yearly_monthly": 160.00,
        "price_label": "$199.99",
        "credits_monthly": 20000,
        "description": "For studios running full production pipelines.",
        "features": [
            "20,000 credits / month",
            "~200 min of animatic video/month",
            "~40 min of full AI video/month",
            "Everything in Pro",
            "Lip-sync & avatar singer shots",
            "Training data pipeline export",
            "Dedicated support",
        ],
        "limits": [],
        "stripe_price_env": "STRIPE_STUDIO_PRICE_ID",
        "stripe_price_yearly_env": None,
        "popular": False,
        "yearly_saving": "20%",
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
    plan_interval = (user or {}).get("plan_interval", "monthly")
    credits = (user or {}).get("credits", 0)
    stripe_enabled = _stripe_enabled()
    return render_template(
        "pricing.html",
        plans=PLANS,
        current_plan=current_plan,
        plan_expires_at=plan_expires_at,
        plan_interval=plan_interval,
        credits=credits,
        stripe_enabled=stripe_enabled,
        credit_costs=CREDIT_COSTS,
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
    plan_interval = request.form.get("interval", "monthly").lower()
    if plan_interval not in ("monthly", "yearly"):
        plan_interval = "monthly"
    if plan_key not in ("starter", "pro", "studio"):
        flash("Invalid plan selection.", "error")
        return redirect(url_for("billing.pricing"))

    price_env = PLANS[plan_key]["stripe_price_env"]
    price_id = os.getenv(price_env) if price_env else None
    if not price_id:
        flash("This plan is not yet available for purchase. Check back soon.", "info")
        return redirect(url_for("billing.pricing"))

    stripe = _stripe()
    user = current_user()
    credits_to_grant = PLANS[plan_key].get("credits_monthly", 0)

    try:
        base_url = request.host_url.rstrip("/")
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            customer_email=user["email"] if not user.get("stripe_customer_id") else None,
            customer=user.get("stripe_customer_id") or None,
            success_url=base_url + url_for("billing.success") + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=base_url + url_for("billing.cancel"),
            metadata={"user_id": str(user["id"]), "plan": plan_key, "interval": plan_interval, "credits": str(credits_to_grant)},
            subscription_data={"metadata": {"user_id": str(user["id"]), "plan": plan_key, "interval": plan_interval}},
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


@billing_bp.route("/billing/portal")
@login_required
def portal():
    """Redirect the user to the Stripe Customer Portal to manage their subscription."""
    stripe = _stripe()
    user = current_user()
    customer_id = (user or {}).get("stripe_customer_id")

    if not stripe or not customer_id:
        flash("No active subscription found. Upgrade from the pricing page.", "info")
        return redirect(url_for("billing.pricing"))

    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=request.host_url.rstrip("/") + url_for("account"),
        )
        return redirect(session.url)
    except Exception as exc:
        logger.error("Stripe portal session error: %s", exc)
        flash("Could not open billing portal — please try again.", "error")
        return redirect(url_for("account"))


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

    plan_interval = meta.get("interval", "monthly")
    credits_to_grant = PLANS.get(plan_key, {}).get("credits_monthly", 0)

    try:
        update_user_plan(
            user_id=int(user_id),
            plan=plan_key,
            stripe_customer_id=customer_id,
            plan_expires_at=expires_at,
            plan_interval=plan_interval,
            credits_to_grant=credits_to_grant,
        )
        logger.info("User %s upgraded to %s (%s) — %d credits granted", user_id, plan_key, plan_interval, credits_to_grant)
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
        plan_key = user.get("plan", "pro")
        credits_to_grant = PLANS.get(plan_key, {}).get("credits_monthly", 0)
        update_user_plan(
            user["id"], plan_key,
            plan_expires_at=expires_at,
            credits_to_grant=credits_to_grant,
        )
    elif status in ("canceled", "unpaid", "past_due"):
        _downgrade_customer(customer_id)


def _downgrade_customer(customer_id: str) -> None:
    user = get_user_by_stripe_customer(customer_id)
    if not user:
        logger.warning("Webhook: no user found for customer %s", customer_id)
        return
    update_user_plan(user["id"], "free", plan_expires_at=None)
    logger.info("Downgraded user %s to free (customer %s)", user["id"], customer_id)
