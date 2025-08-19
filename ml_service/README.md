# health
curl http://localhost:9000/api/healthz

# price
curl -X POST http://localhost:9000/api/v1/price-suggest \
  -H "Content-Type: application/json" \
  -d '{"title":"MacBook Pro 2021","description":"M1, 8/256","category":"Electronics","condition":"Used - Good"}'

# duplicate (single)
curl -X POST http://localhost:9000/api/v1/duplicate-check \
  -H "Content-Type: application/json" \
  -d '{"title":"MacBook Pro 2021 for sale","description":"excellent condition"}'

# duplicate (batch)
curl -X POST http://localhost:9000/api/v1/duplicate-check/batch \
  -H "Content-Type: application/json" \
  -d '{"items":[{"title":"HP 840 G5"},{"title":"Casio fx-991ES"}], "k":10, "threshold":0.75}'

# recommend
curl -X POST http://localhost:9000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"userId":"u_123","campusOnly":true,"k":20}'
Short answer: **yes—Day 6 is essentially done** *if* you can tick off the acceptance checks below. They align with your Day-6 brief (“wire models behind FastAPI; request validation; batching for dup-check; cold-start defaults”) and with the backend’s current API style and modules.

# checks

### Day 6 acceptance checklist (sign-off)

1. **ML FastAPI is running** (Docker or local) with auto OpenAPI (`/docs`, `/openapi.json`) and a **health** route responding OK.
2. **Endpoints live & validated**

   * `POST /api/v1/price-suggest` (Pydantic schema, returns price + confidence + range).
   * `POST /api/v1/duplicate-check` and `POST /api/v1/duplicate-check/batch` (threshold, k, returns similars).
   * `POST /api/v1/recommend` (userId | listingId | campusOnly, returns top-K).
   * All inputs use the same field names the backend uses for listings/auth (to avoid adapters).
3. **Batching for dup-check** path is working and returns results within budget.
4. **Cold-start fallbacks implemented**

   * Price → category×condition median.
   * Recos → campus-aware popularity list.
5. **Latency budgets verified (local/load test)**

   * Price ≤ **300 ms**, Duplicate ≤ **500 ms**, Recos ≤ **300 ms** (p95).
6. **Caching on hot endpoints** (e.g., recos) backed by Redis; keys TTL set (e.g., 10 min).
7. **Backend integration path chosen and wired**

   * Either proxied via backend AI routes **/api/v1/ai/** → ML service, or direct internal calls from backend’s service layer.
   * Env added in backend: `ML_SERVICE_BASE_URL`, request **timeouts/circuit-breaker** set.
8. **Security boundary**

   * ML service only reachable on the private network; JWT stays enforced at backend (consistent with backend’s auth flows).
9. **Basic ops**

   * Structured logs for inputs/outputs (PII-safe), errors.
   * Health check hooked into your monitor.
   * Container image builds cleanly and runs with mounted model artifacts.

If all nine are ✅, then **Day 6 is complete**.

> Context sanity: backend already exposes auth/login/me and listings/search with FastAPI (so mirroring those shapes on your ML side was the right call), and it has a notifications module you can optionally use later for “smart price” or “similar listing” alerts.

**Tiny punch-list that often gets missed (5–10 mins each):**

* Map backend error codes ↔ ML errors (e.g., 422 validation, 504 timeout).
* Add **/healthz** to readiness probes and ensure **/docs** loads in the deployed env.
* Confirm Redis URL matches backend’s infra namespace.
* Save your **OpenAPI JSON** artifact to the repo (helps QA import).

If any one of those is still pending, knock it out and you’re fully done with Day 6. Otherwise, green light to move to **Day 7: Integration & Tuning**.



# Summmary 

Here’s a clear, detailed wrap-up of what you accomplished on **Day 6 – Serve & Scale (ML)** for Campus Exchange.

# Day 6 Summary — What you delivered

## 1) Production-ready ML microservice

* **FastAPI service** exposing three model families behind clean, versioned routes:

  * `POST /api/v1/price-suggest` → price prediction (+ confidence & range).
  * `POST /api/v1/duplicate-check` and `POST /api/v1/duplicate-check/batch` → near-dup detection (top-K, threshold).
  * `POST /api/v1/recommend` → user- and listing-based recommendations with campus filter.
* **Consistent schemas** (Pydantic) that mirror backend listing fields (title, description, category, condition, brand, university), so the backend can call the service without adapters.
* **OpenAPI docs** and **health endpoint** (`/api/healthz`) for checks and QA tooling.

## 2) Models wired with artifacts from Days 3–5

* **Price Suggestion** loads the saved sklearn Pipeline (encoders + regressor), returning a **point estimate**, a **confidence placeholder**, and a **±15% range**.
* **Duplicate Detection** uses the saved text index (TF-IDF matrix & vectorizer; optional image hashes), with an **ensemble-style scorer** and **top-M pruning** for speed.
* **Recommendations** loads ALS+BM25 artifacts (model, mappings, item catalog) and serves:

  * **User→item**: `recommend(userId)`
  * **Listing→listing**: `recommend(listingId)`
  * **Campus-only** re-rank when `campusOnly=true`.

## 3) Fallbacks & resilience (graceful degradation)

* **Price fallback** to `category × condition` median when the model can’t score.
* **Recos fallback** to **campus-aware popularity** for cold-start users/items.
* **Duplicate** returns a helpful recommendation string (e.g., “edit title/description”) when it flags a match.

## 4) Performance & scale controls

* **Latency budgets** (timeboxing decorators):

  * Price ≤ **300 ms**
  * Duplicate ≤ **500 ms**
  * Recos ≤ **300 ms**
* **Batch endpoint** for duplicate checks to support moderation/admin queues efficiently.
* **Caching** via Redis (TTL \~10 min) on hot recommendation queries to keep p95 low.
* **Stateless container** with lazy model loading to keep startup light and per-request latency stable.

## 5) Integration alignment with the backend

* Designed to be **called internally** by the backend (private network), keeping **JWT** and RBAC at the backend boundary.
* Two integration modes supported:

  1. **Proxy through backend** `/api/v1/ai/*` routes → ML service.
  2. **Direct backend service class** calling the ML HTTP endpoints with timeouts & circuit breaker.
* **Environment knobs** provided for:

  * `ML_SERVICE_BASE_URL`, `ML_TIMEOUT_MS`
  * Model paths (price, dup index, reco)
  * `REDIS_URL`, `CACHE_TTL_S`

## 6) Deployability & ops

* **Dockerfile** for reproducible builds.
* **Config module** for all paths, budgets, and infra URLs (12-factor friendly).
* **Structured logging** hooks (timings, outcomes; PII-safe).
* Ready for **readiness/liveness probes** in your host platform (healthz).
* Minimal **observability stubs** (add Sentry/OTel in Day 8 per your plan).

## 7) Validation & smoke tests

* Example **curl** calls for each route (health, price, duplicate single/batch, recommend).
* Schema validation ensures backend payload drift is caught early (422s), with clean error messages.

---

### What changed vs. Day 5

* You moved from notebooks + joblib bundles → **production API** with: contracts, fallbacks, caching, batching, and latency guards.
* RecSys/dup/price are now **callable services**, not just local utilities.

### Known constraints (captured for Day 7/8)

* Confidence for price is a placeholder; proper calibration/intervals can be added later.
* Minimal logging right now (intentionally lightweight) — richer APM/tracing to be added in Day 8.
* Recos “campus-only” relies on item metadata; ensure campus fields are consistently populated.

---

## Day 6 acceptance checklist you hit

* Endpoints live with validated schemas and OpenAPI.
* Batch dup-check implemented.
* Fallbacks for price & recos in place.
* Caching on recos; timeboxing for all three services.
* Containerized; health check ready.
* Integration path & env variables defined.

If you’re happy with this summary, we can jump into **Day 7 – Integration & Tuning** next: wiring the backend http client/proxy, setting explicit timeouts/retries, profiling p95s, and tuning thresholds/factors with quick load tests.




