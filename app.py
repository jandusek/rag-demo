import os, json, base64, ast, math, html, time
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI

# Streamlit configuration
st.set_page_config(page_title="RetailNext Demo", layout="wide")

# Custom CSS for UI styling and card-based layout
st.markdown(
    """
    <style>
    /* Hide file uploader size limit text */
    #root span.st-emotion-cache-gl1nle.eamlidg4 { display: none !important; }
    div[data-testid="stFileUploader"] small { display: none !important; }
    div[data-testid="stFileUploader"] [data-testid="stUploadDropzone"] > div > span:last-child { display: none !important; }
    section [data-testid="stFileUploader"] > div > div > span:last-of-type { display: none !important; }

    /* Hide fullscreen button on images */
    button[title="View fullscreen"] { display: none !important; }
    [data-testid="StyledFullScreenButton"] { display: none !important; }
    [data-testid="stImageFullScreenButton"] { display: none !important; }
    .stImage button { display: none !important; }

    /* Button text formatting */
    div.stButton > button {
        white-space: normal;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    /* Card UI for catalog & recommendations */
    :root { --card-border: rgba(49,51,63,0.2); }
    p { font-weight: 200; }

    a.card {
        display: block;
        text-decoration: none;
        color: inherit;
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,.04);
        transition: transform .06s ease, box-shadow .15s ease;
        overflow: hidden;
        background: inherit;
        margin-bottom: 16px;
    }
    a.card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,.08);
    }
    .card-thumb {
        width: 100%;
        aspect-ratio: 1 / 1;
        object-fit: cover;
        border-radius: 8px;
    }
    .card-title {
        margin-top: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        line-height: 1.2;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        height: calc(2 * 1.2em);
    }
    .card-type {
        margin-top: 4px;
        font-size: 0.85rem;
        line-height: 1.2;
        opacity: 0.75;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        height: 1.2em;
    }

    </style>

    <script>
    // Handle card navigation with page reload
    document.addEventListener('click', function(e) {
        if (e.target.closest('a.card')) {
            e.preventDefault();
            const href = e.target.closest('a.card').href;
            window.location.href = href;
            setTimeout(() => window.location.reload(), 50);
        }
    });
    </script>
    """,
    unsafe_allow_html=True,
)

# Card navigation helpers
@st.cache_data(show_spinner=False)
def image_data_url(style_id: str) -> str:
    p = img_path(style_id)
    if not p:
        return ""
    with open(p, "rb") as f:
        data = f.read()
    return b64_data_url(data, "image/jpeg")


def card_html(sid: str, title: str, type_text: str) -> str:
    img_src = image_data_url(sid)
    title_html = html.escape(title or "")
    type_html = html.escape((type_text or "").title())
    type_html = type_html if type_html else "&nbsp;"
    href = f"?view=detail&id={sid}"
    return f'''
<a class="card" href="{href}" target="_self">
  <img class="card-thumb" src="{img_src}" alt="{title_html}" />
  <div class="card-title">{title_html}</div>
  <div class="card-type">{type_html}</div>
</a>
'''

client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"))

# File paths and configuration
CSV_WITH_EMB = "sample_styles_with_embeddings.csv"   # Primary dataset with precomputed embeddings
CSV_META      = "sample_styles.csv"                  # Fallback metadata-only dataset
IMAGES_DIR    = "sample_images"

# OpenAI models
GPT_MODEL = "gpt-4o-mini"               # For image analysis and text generation
EMBED_MODEL = "text-embedding-3-large"  # For semantic similarity search

# Visual matching configuration
ENABLE_VISUAL_MATCHING = True           # Enable/disable visual outfit matching
MATCH_CHECK_WORKERS = 2                 # Reduced to avoid rate limits

# Data processing utilities
def _find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n in cols: return cols[n]
    return None

def _parse_embedding_cell(x):
    if isinstance(x, (list, tuple, np.ndarray)): return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            try:
                return ast.literal_eval(x)
            except Exception:
                return None
    return None

@st.cache_data(show_spinner=False)
def load_dataframe() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load clothing dataset with consistent column mapping."""
    if os.path.exists(CSV_WITH_EMB):
        df = pd.read_csv(CSV_WITH_EMB)
    else:
        df = pd.read_csv(CSV_META)

    # Map dataset columns to standard names
    mapping = {
        "id": "id",
        "title": "productDisplayName",
        "gender": "gender",
        "type": "articleType",
        "season": "season",
        "color": "baseColour",
        "embed": "embeddings" if "embeddings" in df.columns else None,
    }

    # Ensure consistent data types for caching
    if mapping["embed"]:
        df[mapping["embed"]] = df[mapping["embed"]].astype(str)
    df[mapping["id"]] = df[mapping["id"]].astype(str)

    return df, mapping

@st.cache_resource(show_spinner=False)
def build_matrix(df: pd.DataFrame, embed_col: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Build normalized embedding matrix for efficient cosine similarity."""
    if embed_col is None or embed_col not in df.columns:
        return np.empty((0,)), np.empty((0,))

    # Parse string embeddings to arrays
    parsed = df[embed_col].map(_parse_embedding_cell)
    parsed = parsed.apply(lambda x: _parse_embedding_cell(x) if isinstance(x, str) else x)

    try:
        arr = np.array(parsed.tolist(), dtype=np.float32)
    except Exception:
        return np.empty((0,)), np.empty((0,))

    if arr.size == 0:
        return np.empty((0,)), np.empty((0,))

    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    normalized = arr / norms

    return normalized, norms.squeeze(-1)

def img_path(style_id: str) -> str:
    p = os.path.join(IMAGES_DIR, f"{style_id}.jpg")
    return p if os.path.exists(p) else ""

def b64_data_url(img_bytes: bytes, mime="image/jpeg") -> str:
    return f"data:{mime};base64," + base64.b64encode(img_bytes).decode("utf-8")

def embed_texts(texts: List[str]) -> np.ndarray:
    """Generate normalized embeddings for text queries."""
    if not texts:
        return np.empty((0,))

    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    arr = np.array(vecs, dtype=np.float32)

    # Normalize for cosine similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr / norms

def cosine_topk(query_vecs_n: np.ndarray, corpus_vecs_n: np.ndarray, k=20) -> List[int]:
    """Find top-k most similar items using cosine similarity."""
    if corpus_vecs_n.size == 0 or query_vecs_n.size == 0:
        return []

    # Matrix multiplication for efficient cosine similarity
    similarities = query_vecs_n @ corpus_vecs_n.T
    max_similarities = similarities.max(axis=0)
    top_indices = np.argsort(-max_similarities)[:k]

    return top_indices.tolist()

# Recommendation system utilities

def group_topk_by_type(candidates: pd.DataFrame, type_col: Optional[str], sim_scores: np.ndarray,
                        per_type: int = 2, exclude_ids: Optional[set] = None) -> pd.DataFrame:
    """Group recommendations by item type to ensure diversity."""
    if candidates.empty:
        return candidates

    # Add similarity scores and sort
    tmp = candidates.copy()
    tmp["_score"] = sim_scores[: len(tmp)] if sim_scores is not None else 0.0

    # Track items per type for diversity
    seen_per_type: Dict[str, int] = {}
    selected_rows = []

    for _, row in tmp.sort_values("_score", ascending=False).iterrows():
        row_id = str(row[COL["id"]])

        # Skip excluded items
        if exclude_ids and row_id in exclude_ids:
            continue

        # Normalize type for grouping
        item_type = str(row[type_col]).strip().lower() if type_col else "other"

        # Skip if we have enough of this type
        if seen_per_type.get(item_type, 0) >= per_type:
            continue

        selected_rows.append(row)
        seen_per_type[item_type] = seen_per_type.get(item_type, 0) + 1

    return pd.DataFrame(selected_rows)


def find_similar_items(query_text: str, pool_idx: np.ndarray, k: int = 2) -> pd.DataFrame:
    """Find k most similar items for a single query text using cosine similarity."""
    if not query_text:
        return pd.DataFrame()

    # Generate embedding for single query
    query_embedding = embed_texts([query_text])
    if query_embedding.size == 0:
        return pd.DataFrame()

    # Find most similar items in the filtered pool
    local_indices = cosine_topk(query_embedding, VEC_N[pool_idx], k=k)
    selected_indices = pool_idx[local_indices]
    candidates = df.iloc[selected_indices].copy()

    # Calculate similarity scores
    similarities = (query_embedding @ VEC_N[selected_indices].T).flatten()
    candidates["_score"] = similarities
    candidates["_query"] = query_text

    return candidates.sort_values("_score", ascending=False)

def find_matching_items_with_rag(item_descriptions: List[str], pool_idx: np.ndarray,
                                reference_image_bytes: Optional[bytes] = None,
                                enable_matching: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """Find matching items using RAG approach - runs cosine search per item description."""
    if not item_descriptions:
        return pd.DataFrame(), {}

    all_candidates = []

    # Process each item description individually (3 items → 6 results total)
    for description in item_descriptions:
        similar_items = find_similar_items(description, pool_idx, k=2)
        if not similar_items.empty:
            all_candidates.append(similar_items)

    if not all_candidates:
        return pd.DataFrame(), {}

    # Combine all results
    combined_candidates = pd.concat(all_candidates, ignore_index=True)

    # Apply visual matching filter if enabled and reference image is provided
    matching_debug_info = {}
    if enable_matching and reference_image_bytes is not None:
        with st.spinner("Checking outfit compatibility..."):
            filtered_candidates, matching_debug_info = filter_recommendations_with_matching(reference_image_bytes, combined_candidates)
            return filtered_candidates, matching_debug_info

    return combined_candidates, matching_debug_info

def _recs_with_spinner(titles: List[str], pool_idx: np.ndarray, reference_image_bytes: Optional[bytes] = None, enable_matching: bool = True):
    """Compute recommendations with loading indicator using RAG approach."""
    with st.spinner("Finding complementary items…"):
        return find_matching_items_with_rag(titles, pool_idx, reference_image_bytes=reference_image_bytes, enable_matching=enable_matching)

def analyze_image_with_gpt(image_bytes: bytes, subcategories: List[str]) -> Dict:
    """Analyze clothing image and generate complementary item recommendations."""
    data_url = b64_data_url(image_bytes)

    # Enhanced prompt based on the original cookbook
    prompt = (
        "Given an image of an item of clothing, analyze the item and generate a JSON output with the following fields: \"items\", \"category\", and \"gender\"."
        "Use your understanding of fashion trends, styles, and gender preferences to provide accurate and relevant suggestions for how to complete the outfit."
        "The items field should be a list of items that would go well with the item in the picture. Each item should represent a title of an item of clothing that contains the style, color, and gender of the item."
        f"The category needs to be chosen between the types in this list: {sorted(set(subcategories))}\n."
        "You have to choose between the genders in this list: [Men, Women, Boys, Girls, Unisex]"
        "Do not include the description of the item in the picture. Do not include the ```json ``` tag in the output."
        "Example Input: An image representing a black leather jacket."
        'Example Output: {{"items": ["Fitted White Women\'s T-shirt", "White Canvas Sneakers", "Women\'s Black Skinny Jeans"], "category": "Jackets", "gender": "Women"}}'
    )

    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }],
        temperature=0.3  # Lower temperature for more consistent results
    )

    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {
            "items": [],
            "category": None,
            "gender": None,
            "description": resp.choices[0].message.content
        }

def generate_marketing_copy(item_title: str, attrs: Dict, image_bytes: Optional[bytes]) -> str:
    """Generate marketing copy for clothing items."""
    content = [{
        "type": "text",
        "text": (
            "Write a compelling product description (≤60 words) that highlights:\n"
            "- Style and aesthetic appeal\n"
            "- Fit and comfort\n"
            "- Fabric quality and feel\n"
            "- Seasonal appropriateness\n"
            "- Styling versatility\n\n"
            f"Product: {item_title}\n"
            f"Attributes: {attrs}"
        )
    }]

    if image_bytes:
        content.append({
            "type": "image_url",
            "image_url": {"url": b64_data_url(image_bytes)}
        })

    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0.7  # Higher creativity for marketing copy
    )

    return resp.choices[0].message.content.strip()

def check_match(reference_image_base64: str, suggested_image_base64: str) -> Dict:
    """Check if two clothing items would work well together in an outfit."""
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """ You will be given two images of two different items of clothing.
                            Your goal is to decide if the items in the images would work in an outfit together.
                            The first image is the reference item (the item that the user is trying to match with another item).
                            You need to decide if the second item would work well with the reference item, aim for 50% rejection rate.
                            Your response must be a JSON output with the following fields: "answer", "reason".
                            The "answer" field must be either "yes" or "no", depending on whether you think the items would work well together.
                            The "reason" field must be a short explanation of your reasoning for your decision. Do not include the descriptions of the 2 images.
                            Do not include the ```json ``` tag in the output.
                        """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{reference_image_base64}",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{suggested_image_base64}",
                        },
                    }
                ],
            }
        ],
        max_tokens=300,
        temperature=0.1  # Low temperature for consistent matching decisions
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception:
        # Fallback if JSON parsing fails
        return {
            "answer": "no",
            "reason": "Unable to parse matching response"
        }

def check_single_match(reference_image_b64: str, candidate_item: pd.Series) -> Tuple[str, bool, str]:
    """Check if a single candidate item matches the reference image."""
    candidate_id = str(candidate_item[COL["id"]])
    candidate_img_path = img_path(candidate_id)

    if not candidate_img_path:
        return candidate_id, False, "No image available"

    try:
        # Load and encode candidate image
        with open(candidate_img_path, "rb") as f:
            candidate_img_bytes = f.read()
        candidate_img_b64 = base64.b64encode(candidate_img_bytes).decode("utf-8")

        # Check match using GPT (with small delay to avoid rate limits)
        time.sleep(0.3)
        match_result = check_match(reference_image_b64, candidate_img_b64)
        is_match = match_result.get("answer", "no").lower() == "yes"
        reason = match_result.get("reason", "Unknown")

        return candidate_id, is_match, reason

    except Exception as e:
        return candidate_id, False, f"Error processing: {str(e)}"

def filter_recommendations_with_matching(reference_image_bytes: bytes, candidates: pd.DataFrame, max_workers: int = MATCH_CHECK_WORKERS) -> Tuple[pd.DataFrame, Dict]:
    """Filter recommendations using parallelized outfit matching."""
    if candidates.empty:
        return candidates, {"total_checked": 0, "passed": 0, "failed": 0, "details": []}

    # Encode reference image
    reference_img_b64 = base64.b64encode(reference_image_bytes).decode("utf-8")

    # Parallel processing of match checks
    matching_items = []
    match_details = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all match checks
        future_to_item = {
            executor.submit(check_single_match, reference_img_b64, row): idx
            for idx, (_, row) in enumerate(candidates.iterrows())
        }

        # Collect results as they complete
        for future in as_completed(future_to_item):
            item_idx = future_to_item[future]
            try:
                item_id, is_match, reason = future.result()

                # Get item details for debugging
                item_row = candidates.iloc[item_idx]
                item_details = {
                    "item_id": item_id,
                    "item_title": str(item_row[COL["title"]]),
                    "item_type": str(item_row[COL["type"]]) if COL["type"] else "Unknown",
                    "item_gender": str(item_row[COL["gender"]]) if COL["gender"] else "Unknown",
                    "similarity_score": float(item_row.get("_score", 0.0)),
                    "is_match": is_match,
                    "match_reason": reason,
                    "original_idx": item_idx
                }
                match_details.append(item_details)

                if is_match:
                    matching_items.append(item_idx)

            except Exception as e:
                st.warning(f"Error checking match for item: {str(e)}")

    # Create debug summary
    debug_info = {
        "total_checked": len(match_details),
        "passed": len([d for d in match_details if d["is_match"]]),
        "failed": len([d for d in match_details if not d["is_match"]]),
        "details": sorted(match_details, key=lambda x: x["similarity_score"], reverse=True)
    }

    # Filter candidates to only matching items
    if matching_items:
        matching_candidates = candidates.iloc[matching_items].copy()
        # Add match details for debugging
        matching_candidates["_match_reason"] = [
            next(d["match_reason"] for d in match_details
                 if d["original_idx"] == idx and d["is_match"])
            for idx in matching_items
        ]
        return matching_candidates, debug_info
    else:
        return pd.DataFrame(), debug_info  # Return empty DataFrame if no matches


def check_item_compatibility(reference_image_bytes: bytes, item_id: str) -> bool:
    """Quick compatibility check for a single item."""
    try:
        candidate_img_path = img_path(item_id)
        if not candidate_img_path:
            return False

        # Encode reference image
        reference_img_b64 = base64.b64encode(reference_image_bytes).decode("utf-8")

        # Load and encode candidate image
        with open(candidate_img_path, "rb") as f:
            candidate_img_bytes = f.read()
        candidate_img_b64 = base64.b64encode(candidate_img_bytes).decode("utf-8")

        # Check match
        match_result = check_match(reference_img_b64, candidate_img_b64)
        return match_result.get("answer", "no").lower() == "yes"

    except Exception:
        return False

# Data initialization
with st.spinner("Loading catalog..."):
    df, COL = load_dataframe()
    VEC_N, _ = build_matrix(df, COL["embed"])  # Pre-normalized embeddings for efficient search

# Navigation utilities
def qp_get(key: str):
    try:
        return st.query_params.get(key)
    except Exception:
        return None

def render_detail_page(item_id: Optional[str] = None, is_user_upload: bool = False):
    # Back to catalog
    if st.button("← Back to catalog"):
        # Clear upload-related session state when going back
        for key in ["upload_image_b64", "upload_analysis", "pending_upload_bytes"]:
            if key in st.session_state:
                del st.session_state[key]

        st.query_params.clear()
        st.rerun()

    # Helpers to normalize
    def norm_gender(s):
        return str(s).strip().title()
    def norm_type(s):
        return str(s).strip().lower()

    # We will compute analysis + filters, then build pool
    analysis = {}
    pool = df.copy()
    recs = pd.DataFrame()

    if is_user_upload:
        up_b64 = st.session_state.get("upload_image_b64")
        analysis = st.session_state.get("upload_analysis", {}) or {}
        if not up_b64:
            st.warning("No uploaded image found in the session. Go back and upload again.")
            return
        up_bytes = base64.b64decode(up_b64)
        c1, c2 = st.columns([0.35, 0.65])
        with c1:
            st.image(up_bytes, caption="User-uploaded image", use_container_width=True)
        with c2:
            st.subheader("User item")
            if analysis.get("description"):
                st.write(analysis["description"])
        with st.expander("[debug] Model analysis", expanded=False):
            st.json(analysis)

        # Build filters from model analysis
        pool_before = len(pool)
        if COL["gender"] and analysis.get("gender"):
            target_g = norm_gender(analysis["gender"]) or None
            if target_g:
                allowed = [target_g]
                if target_g != "Unisex":
                    allowed.append("Unisex")
                pool = pool[ pool[COL["gender"]].map(norm_gender).isin(allowed) ]
        pool_after_gender = len(pool)
        if COL["type"] and analysis.get("category"):
            det_t = norm_type(analysis["category"]) or None
            if det_t:
                pool = pool[ pool[COL["type"]].map(norm_type) != det_t ]
        pool_after_type = len(pool)
        pool_idx = pool.index.values

        titles = analysis.get("items", [])
        if not titles:
            st.info("Model returned no complementary item titles; try another image.")
            return
        # Cache key for recommendations
        cache_key = f"upload_recs_{hash(tuple(titles))}"

        # Check if we have cached recommendations
        if cache_key in st.session_state:
            recs = st.session_state[cache_key]
            candidates = st.session_state.get(f"{cache_key}_candidates", pd.DataFrame())
        else:
            # Generate initial recommendations using RAG approach (3 items → 6 results)
            candidates, debug_info = _recs_with_spinner(titles, pool_idx)
            recs = group_topk_by_type(candidates, COL["type"], candidates["_score"].to_numpy() if not candidates.empty else np.array([]), per_type=2)
            # Cache the results
            st.session_state[cache_key] = recs
            # Cache candidates for debugging
            st.session_state[f"{cache_key}_candidates"] = candidates

        with st.expander("[debug] Recommendations", expanded=False):
            st.write({
                "pool_before": pool_before,
                "pool_after_gender": pool_after_gender,
                "pool_after_type": pool_after_type,
                "query_titles": titles,
            })
            if not candidates.empty:
                debug_cols = [COL["id"], COL["title"], COL["gender"], COL["type"], "_score"]
                if "_query" in candidates.columns:
                    debug_cols.append("_query")
                if "_match_reason" in candidates.columns:
                    debug_cols.append("_match_reason")
                preview = candidates[debug_cols].head(20)
                st.dataframe(preview, use_container_width=True)
            if not recs.empty and "_match_reason" in recs.columns:
                st.write("Visual matching applied - showing only compatible items")


    else:
        item_df = df[df[COL["id"]] == str(item_id)]
        if item_df.empty:
            st.error("Item not found.")
            return
        item = item_df.iloc[0]
        img_p = img_path(str(item[COL["id"]]))

        dcols = st.columns([0.35, 0.65])
        with dcols[0]:
            if img_p: st.image(img_p, use_container_width=True)
            else: st.write("(no image)")
        with dcols[1]:
            st.subheader(str(item[COL["title"]]))
            if COL["gender"]: st.write(f"Gender: **{str(item[COL['gender']]).title()}**")
            if COL["type"]:   st.write(f"Item Type: **{str(item[COL['type']]).title()}**")
            if COL["season"]: st.write(f"Season: **{str(item[COL['season']]).title()}**")
            if COL["color"]:  st.write(f"Base Color: **{str(item[COL['color']]).title()}**")
            if st.button("Generate marketing copy"):
                img_bytes = open(img_p, "rb").read() if img_p else None
                attrs = {k: item[COL[k]] for k in ["gender","type","season","color"] if COL[k]}
                copy = generate_marketing_copy(str(item[COL["title"]]), attrs, img_bytes)
                st.success(copy)

        subcats = list(df[COL["type"]].dropna().unique()) if COL["type"] else []
        if img_p and os.path.exists(img_p):
            with open(img_p, "rb") as f:
                img_bytes = f.read()
            with st.spinner("Analyzing the item image…"):
                analysis = analyze_image_with_gpt(img_bytes, subcats)
        else:
            analysis = {"items": [], "description": ""}

        with st.expander("[debug] Model analysis", expanded=False):
            st.json(analysis)

        pool_before = len(pool)
        if COL["gender"] and pd.notna(item.get(COL["gender"])):
            target_g = norm_gender(item[COL["gender"]])
            allowed = [target_g]
            if target_g != "Unisex":
                allowed.append("Unisex")
            pool = pool[ pool[COL["gender"]].map(norm_gender).isin(allowed) ]
        pool_after_gender = len(pool)
        if COL["type"] and pd.notna(item.get(COL["type"])):
            this_t = norm_type(item[COL["type"]])
            pool = pool[ pool[COL["type"]].map(norm_type) != this_t ]
        pool_after_type = len(pool)
        pool_idx = pool.index.values

        titles = analysis.get("items", [])
        exclude_ids = {str(item[COL["id"]])}
        # Generate and filter recommendations with visual matching
        reference_img_bytes = None
        if img_p:
            with open(img_p, "rb") as f:
                reference_img_bytes = f.read()

        # Cache key for recommendations
        cache_key = f"catalog_recs_{item_id}_{hash(tuple(titles))}"

        # Check if we have cached recommendations
        if cache_key in st.session_state:
            recs = st.session_state[cache_key]
            candidates = st.session_state.get(f"{cache_key}_candidates", pd.DataFrame())
        else:
            # Generate initial recommendations using RAG approach (3 items → 6 results)
            candidates, debug_info = _recs_with_spinner(titles, pool_idx)
            recs = group_topk_by_type(
                candidates, COL["type"], candidates["_score"].to_numpy() if not candidates.empty else np.array([]),
                per_type=2, exclude_ids=exclude_ids
            )
            # Cache the results
            st.session_state[cache_key] = recs

        with st.expander("[debug] Recommendations", expanded=False):
            st.write({
                "ref_gender": norm_gender(item[COL["gender"]]) if COL["gender"] else None,
                "ref_type": norm_type(item[COL["type"]]) if COL["type"] else None,
                "pool_before": pool_before,
                "pool_after_gender": pool_after_gender,
                "pool_after_type": pool_after_type,
                "query_titles": titles,
            })
            if not candidates.empty:
                debug_cols = [COL["id"], COL["title"], COL["gender"], COL["type"], "_score"]
                if "_query" in candidates.columns:
                    debug_cols.append("_query")
                if "_match_reason" in candidates.columns:
                    debug_cols.append("_match_reason")
                preview = candidates[debug_cols].head(20)
                st.dataframe(preview, use_container_width=True)
            if not recs.empty and "_match_reason" in recs.columns:
                st.write("Visual matching applied - showing only compatible items")


    # Display recommendations
    st.subheader("Complete the look")
    if recs.empty:
        st.info("No recommendations found. Try a different image or adjust filters.")
        return

    # Store current recommendations and filtering results
    current_recs = recs.copy()
    filtering_results = None

    # Visual filtering button
    ref_img_bytes = up_bytes if is_user_upload else reference_img_bytes
    if ref_img_bytes and st.button("🔍 Advanced AI Stylist Filter"):
        # Run the RAG approach with visual matching enabled
        titles = analysis.get("items", [])
        pool_idx = pool.index.values if is_user_upload else pool.index.values

        filtered_candidates, filtering_results = find_matching_items_with_rag(
            titles, pool_idx, reference_image_bytes=ref_img_bytes, enable_matching=True
        )

        if not filtered_candidates.empty:
            exclude_ids = set() if is_user_upload else {str(item[COL["id"]])} if 'item' in locals() else set()
            current_recs = group_topk_by_type(
                filtered_candidates, COL["type"],
                filtered_candidates["_score"].to_numpy(),
                per_type=2, exclude_ids=exclude_ids
            )
            st.success(f"✅ {len(current_recs)} items hand-picked by your personal AI stylist using RAG approach.")
        else:
            st.warning("❌ No items passed visual compatibility check. Showing all recommendations.")

    # Render recommendations
    recommendation_cols = st.columns(6)
    for idx, (_, row) in enumerate(current_recs.iterrows()):
        col = recommendation_cols[idx % 6]
        item_id = str(row[COL["id"]])
        item_title = str(row[COL["title"]])
        item_type = str(row[COL["type"]]) if COL["type"] else ""
        col.markdown(card_html(item_id, item_title, item_type), unsafe_allow_html=True)

    # Debug info for filtering results
    if filtering_results:
        with st.expander("[debug] Visual Filtering Results", expanded=False):
            st.write(f"**Summary:** {filtering_results['passed']}/{filtering_results['total_checked']} items passed")

            # Create results table
            debug_data = []
            for detail in filtering_results["details"]:
                debug_data.append({
                    "Item ID": detail["item_id"],
                    "Title": detail["item_title"],
                    "Type": detail["item_type"],
                    "Result": "✅ ACCEPTED" if detail["is_match"] else "❌ REJECTED",
                    "AI Reasoning": detail["match_reason"]
                })

            st.dataframe(pd.DataFrame(debug_data), use_container_width=True)

# ---------- Main (catalog) page ----------

def render_catalog_page():
    # Sidebar filters
    st.sidebar.header("Filters")
    def _opts(colname: Optional[str]) -> List[str]:
        if not colname: return []
        vals = sorted([v for v in df[colname].dropna().astype(str).unique() if v != ""])
        return vals

    gender = st.sidebar.multiselect("Gender", _opts(COL["gender"]))
    itype  = st.sidebar.multiselect("Item type", _opts(COL["type"]))
    season = st.sidebar.multiselect("Season", _opts(COL["season"]))
    color  = st.sidebar.multiselect("Base color", _opts(COL["color"]))
    st.sidebar.divider()

    # Filtering
    mask = pd.Series(True, index=df.index)
    if gender and COL["gender"]: mask &= df[COL["gender"]].astype(str).isin(gender)
    if itype  and COL["type"]:   mask &= df[COL["type"]].astype(str).isin(itype)
    if season and COL["season"]: mask &= df[COL["season"]].astype(str).isin(season)
    if color  and COL["color"]:  mask &= df[COL["color"]].astype(str).isin(color)

    view = df[mask].copy()

    # Header & Upload box
    left, right = st.columns([0.65, 0.35])
    with left:
        st.title("RetailNext\nOutfit Recommendation System Demo")
        st.caption("Click any item for details and styling suggestions, or upload your own clothing photo for personalized recommendations.")

    with right:
        st.subheader("Upload a reference item")

        # If an upload is pending processing, show spinner and process here
        pending_bytes = st.session_state.get("pending_upload_bytes")
        if pending_bytes is not None:
            with st.spinner("Analyzing your item & generating recommendations…"):
                img_bytes = pending_bytes
                subcats = list(df[COL["type"]].dropna().unique()) if COL["type"] else []
                analysis = analyze_image_with_gpt(img_bytes, subcats)
                st.session_state["upload_image_b64"] = base64.b64encode(img_bytes).decode("utf-8")
                st.session_state["upload_analysis"] = analysis
                st.session_state["pending_upload_bytes"] = None  # clear
                st.query_params["view"] = "detail"
                st.query_params["user"] = "1"
                st.rerun()

        up = st.file_uploader(
            "Recommend apparel to complete your own items",
            type=["jpg","jpeg","png","webp"]
        )
        if up is not None:
            # Store bytes and rerun so the spinner shows during a full render
            st.session_state["pending_upload_bytes"] = up.read()
            st.rerun()

    # Catalog grid with pagination (controls on the top-right)
    top_l, top_r = st.columns([0.72, 0.28])
    with top_l:
        st.subheader(f"Catalog ({len(view)} items)")
    with top_r:
        per_page = 36
        n_pages = max(1, math.ceil(len(view)/per_page))
        page = int(st.session_state.get("page", 1))
        # Clamp page if filters changed
        if page > n_pages:
            page = n_pages
            st.session_state["page"] = page
        if page < 1:
            page = 1
            st.session_state["page"] = page
        pg_prev, pg_label, pg_next = st.columns([0.25, 0.5, 0.25])
        with pg_prev:
            if st.button("‹", use_container_width=True, disabled=(page <= 1), key="pg_prev"):
                st.session_state["page"] = max(1, page - 1)
                st.rerun()
        with pg_label:
            st.markdown(f"<div style='text-align:center; padding:8px 0;'>Page {page} / {n_pages}</div>", unsafe_allow_html=True)
        with pg_next:
            if st.button("›", use_container_width=True, disabled=(page >= n_pages), key="pg_next"):
                st.session_state["page"] = min(n_pages, page + 1)
                st.rerun()

    start = (page-1)*per_page
    page_df = view.iloc[start:start+per_page]

    # Grid using HTML cards
    cols = st.columns(6)
    for i, (_, row) in enumerate(page_df.iterrows()):
        c = cols[i % 6]
        sid = str(row[COL["id"]])
        title = str(row[COL["title"]])
        ttext = str(row[COL["type"]]) if COL["type"] else ""
        c.markdown(card_html(sid, title, ttext), unsafe_allow_html=True)

# Application routing
view_param = qp_get("view")
item_id_param = qp_get("id")
user_upload_param = qp_get("user")

# Route to appropriate page
if view_param == "detail":
    render_detail_page(item_id_param, is_user_upload=(user_upload_param == "1"))
else:
    render_catalog_page()
