# NutriGraph — Deployment Guide

Two web apps + one Neo4j database, on `abhinavsikhwal.com` subdomains.

| Piece | Subdomain | What it is |
|---|---|---|
| Showcase site | `nutri.abhinavsikhwal.com` | The narrative portfolio page (static) |
| Graph explorer | `graph.nutri.abhinavsikhwal.com` | The live interactive Neo4j graph |
| Neo4j database | internal (`bolt://localhost:7687`) | Backs the graph explorer |

You can ship the **showcase first** (it's fully static, no backend) and add the graph once Neo4j is up.

---

## PART 1 — The showcase site (do this first, it's the quick win)

The showcase is a static site. Build it once, serve the files.

### 1a. Build it (on your machine or the VPS)
```bash
cd web/nutri-showcase
npm install
npm run build
```
This produces a `dist/` folder with plain static HTML/JS/CSS.

### 1b. Serve it with nginx
Copy `dist/` to the VPS, e.g. `/var/www/nutri/`, then an nginx server block:
```nginx
server {
    listen 80;
    server_name nutri.abhinavsikhwal.com;
    root /var/www/nutri;
    index index.html;
    location / { try_files $uri $uri/ /index.html; }
}
```
```bash
sudo ln -s /etc/nginx/sites-available/nutri /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### 1c. DNS + HTTPS
- Add an **A record**: `nutri` → your VPS IP (in your domain's DNS panel).
- Get free HTTPS:
```bash
sudo certbot --nginx -d nutri.abhinavsikhwal.com
```

**Done — the showcase is live.** You can share it now.

---

## PART 2 — Neo4j database (needed only for the graph explorer)

### 2a. Install Neo4j Community
```bash
# Ubuntu/Debian
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/neo4j.gpg
echo "deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable latest" | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update && sudo apt install neo4j -y
sudo systemctl enable neo4j && sudo systemctl start neo4j
```
First login is at `http://<vps-ip>:7474` (user `neo4j`, default password `neo4j`, it forces a change).

### 2b. Load your graph
Copy your `neo4j_load/` folder (from `data/processed/product/neo4j_load/`, produced by `scripts/product/build_neo4j_trimmed_graph.py`) to the VPS. Then either:

**Option A — the Python loader (simplest):**
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="<your-admin-password>"
python scripts/product/build_neo4j_trimmed_graph.py --load
```

**Option B — LOAD CSV:** put the CSVs in Neo4j's import dir and run `load_csv.cypher` in the Neo4j Browser. (See `NEO4J_SETUP.md` from the load package.)

Then create the indexes (from `NEO4J_SETUP.md`).

### 2c. Create the READ-ONLY user (this is what makes public exploration safe)
In the Neo4j Browser (`http://<vps-ip>:7474`), run:
```cypher
CREATE USER readonly IF NOT EXISTS
  SET PASSWORD 'PICK_A_STRONG_READONLY_PASSWORD'
  SET PASSWORD CHANGE NOT REQUIRED;
GRANT ROLE reader TO readonly;
```
The `readonly` user can run any `MATCH` but physically cannot write or delete. The graph frontend connects as this user.

### 2d. Expose Bolt for the browser to connect
The graph frontend (Neo4j JS driver) connects over Bolt from the visitor's browser, so Bolt must be reachable. Two ways:

- **Simplest:** open port 7687 and connect to `bolt://graph.nutri.abhinavsikhwal.com:7687`. Fine for a demo with the read-only user.
- **Recommended (secure, encrypted):** put Bolt behind nginx with TLS so the browser uses `bolt+s://` (encrypted). Add DNS A record `graph.nutri` → VPS IP, get a cert with certbot for it, and proxy 7687. (Neo4j docs: "Configure a reverse proxy" / "Bolt over TLS".)

> If you'd rather not expose Bolt at all, the alternative is a **thin read-only API** (a small Node/Python server that runs the parameterized Cypher and returns JSON; the browser talks to the API, not Neo4j). This is the more hardened public setup. See also `data/processed/product/neo4j_load/NEO4J_SETUP.md` §7.

---

## PART 3 — The graph explorer

### 3a. Wire it to your live Neo4j
Open `web/nutri-graph/src/App.jsx`. At the top, fill the `NEO4J` config:
```js
const NEO4J = {
  uri: "bolt+s://graph.nutri.abhinavsikhwal.com:7687", // or bolt:// for unencrypted
  user: "readonly",
  password: "YOUR_READONLY_PASSWORD",
};
```

### 3b. Swap the sample data for live queries
The file currently runs on an embedded sample so it works offline. To go live, replace the three sample calls with the Neo4j driver. Search for these functions in `App.jsx`: `search`, `seed`, `expand`. Each has a marked line like `// live: run cypher ...`.

Add near the top of the file:
```js
import neo4j from "neo4j-driver";
const driver = NEO4J.uri
  ? neo4j.driver(NEO4J.uri, neo4j.auth.basic(NEO4J.user, NEO4J.password))
  : null;

async function runCypher(cypher, params) {
  const session = driver.session({ defaultAccessMode: neo4j.session.READ });
  try {
    const res = await session.run(cypher, params);
    return res.records;
  } finally { await session.close(); }
}
```
Then replace `sampleSearch(q)` with a call to the search template, and `sampleNeighbors(id, ev)` with the expand template (both are in `cypher_templates.json` from the load package). Map the returned records into the same `{nodes, edges}` shape the sample uses; the rendering code stays identical.

### 3c. Build and serve
```bash
cd web/nutri-graph
npm install
npm run build
```
Serve `dist/` under `graph.nutri.abhinavsikhwal.com` with the same nginx + certbot pattern as Part 1.

---

## Recommended order
1. **Ship the showcase** (Part 1) — live and shareable today, no backend.
2. Point the showcase's "Explore the live graph" button at `https://graph.nutri.abhinavsikhwal.com`.
3. Stand up **Neo4j + read-only user** (Part 2).
4. Wire and ship the **graph explorer** (Part 3).

## Notes
- Both apps have zero paid dependencies. Free fonts via Google Fonts.
- Free HTTPS via certbot/Let's Encrypt.
- The read-only Neo4j user is the safety boundary: keep the admin password private; only ship the `readonly` credentials in the frontend (never commit a real password — replace deploy-time placeholders such as `CHANGE_ME_READONLY_PASSWORD` / `YOUR_READONLY_PASSWORD` on the server only).
- For a hardened public graph, prefer the thin read-only API (see `NEO4J_SETUP.md`) instead of exposing Bolt directly.
