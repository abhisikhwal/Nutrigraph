# NutriGraph API + Live Frontend, VPS setup

## 1. The read-only API (server.js)
Put `server.js` + `package.json` in `~/nutri-api/` on the VPS.

```bash
cd ~/nutri-api
npm install
```

Run it with your Neo4j credentials in the environment. Use the ADMIN
neo4j user (Community edition has no read-only role; the API enforces
read-only by only running fixed read queries):

```bash
NEO4J_URI="bolt://localhost:7687" \
NEO4J_USER="neo4j" \
NEO4J_PASSWORD="YOUR_ADMIN_PASSWORD" \
NUTRI_API_PORT=8600 \
node server.js
```

Test locally:
```bash
curl "http://127.0.0.1:8600/health"
curl "http://127.0.0.1:8600/search?q=turmeric"
```

To keep it running after logout, use pm2 (like the hotel bot) or a
systemd service. pm2 example:
```bash
sudo npm install -g pm2   # if not already installed
NEO4J_PASSWORD="YOUR_ADMIN_PASSWORD" pm2 start server.js --name nutri-api
pm2 save
```

## 2. The frontend
Replace `web/nutri-graph/src/App.jsx` in the repo with `App_live.jsx`
(rename it to App.jsx). It calls the API at `/api` (same domain, Caddy
proxies /api to the API server, see Caddy block below).

```bash
cd ~/Nutrigraph/web/nutri-graph
npm install
npm run build
```

## 3. Caddy block for graph.nutri.abhinavsikhwal.com
Serves the static graph page AND proxies /api to the API server:

```
graph.nutri.abhinavsikhwal.com {
        handle /api/* {
                uri strip_prefix /api
                reverse_proxy 127.0.0.1:8600
        }
        handle {
                root * /home/abhinav/Nutrigraph/web/nutri-graph/dist
                try_files {path} /index.html
                file_server
        }
}
```
