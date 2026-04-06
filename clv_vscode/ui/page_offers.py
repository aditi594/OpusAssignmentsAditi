import streamlit as st
from models.offer_manager import load_offers, add_offer, update_offer, delete_offer

def render():
    st.header("🎯 Personalized Offer Management")

    offers = load_offers()

    # ─────────────────────────────────────────
    # Existing Offers
    # ─────────────────────────────────────────
    st.subheader("Existing Offers")

    if not offers:
        st.info("No offers created yet.")
    else:
        for o in offers:
            with st.container():
                col1, col2, col3, col4 = st.columns([4,1,1,1])

                # Offer details
                status = "✅ Active" if o.get("active", True) else "⛔ Inactive"
                col1.markdown(
                    f"""
                    **{o["title"]}**  
                    _Segment_: {o["segment"]}  
                    _Category_: {o["category"]}  
                    _Status_: {status}  

                    {o["description"]}
                    """
                )

                # Edit
                if col2.button("✏ Edit", key=f"edit_{o['id']}"):
                    st.session_state.edit_offer = o

                # Activate / Deactivate
                if col3.button(
                    "Disable" if o.get("active", True) else "Enable",
                    key=f"toggle_{o['id']}"
                ):
                    update_offer(o["id"], {"active": not o.get("active", True)})
                    st.rerun()

                # Delete
                if col4.button("🗑 Delete", key=f"del_{o['id']}"):
                    delete_offer(o["id"])
                    st.rerun()

    st.divider()

    # ─────────────────────────────────────────
    # Edit Offer
    # ─────────────────────────────────────────
    if "edit_offer" in st.session_state:
        o = st.session_state.edit_offer
        st.subheader("Edit Offer")

        with st.form("edit_offer_form"):
            title = st.text_input("Offer Title", o["title"])
            description = st.text_area("Description", o["description"])
            category = st.text_input("Category", o["category"])
            segment = st.selectbox(
                "Customer Segment",
                ["High", "Medium", "Low"],
                index=["High","Medium","Low"].index(o["segment"])
            )

            submitted = st.form_submit_button("Update Offer")

        if submitted:
            update_offer(
                o["id"],
                {
                    "title": title,
                    "description": description,
                    "category": category,
                    "segment": segment,
                }
            )
            del st.session_state.edit_offer
            st.success("Offer updated successfully")
            st.rerun()

        if st.button("Cancel Edit"):
            del st.session_state.edit_offer
            st.rerun()

    # ─────────────────────────────────────────
    # Add New Offer
    # ─────────────────────────────────────────
    st.subheader("Add New Offer")

    with st.form("add_offer"):
        segment = st.selectbox("Customer Segment", ["High", "Medium", "Low"])
        title = st.text_input("Offer Title")
        description = st.text_area("Description")
        category = st.text_input("Category")

        submitted = st.form_submit_button("Add Offer")

    if submitted:
        if not title or not description or not category:
            st.error("All fields are required.")
        else:
            add_offer(segment, title, description, category)
            st.success("Offer added successfully")
            st.rerun()
