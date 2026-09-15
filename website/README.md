# rGPU product documentation

A standalone Next.js / Fumadocs MDX site. It does not change the Python or CUDA
packages and does not need a running GPU server.

## Run locally

Use Node.js 22 or newer and npm. The initial build was verified with Node 26.

```sh
cd website
npm ci
npm run dev
```

Open http://localhost:3000. Edit `content/docs/*.mdx`; navigation order lives in
`content/docs/meta.json`. The landing page is `app/(home)/page.tsx`.

## Validate and preview the static build

```sh
npm run build
npm run types:check
npm start
```

`build` compiles MDX, checks TypeScript and exports the site to `out/`. `start`
serves that directory locally. Search uses a generated static index; no hosted
search service, API keys, or server runtime are required. Markdown copies and
`/llms.txt` are generated from the same content.

Deploy the contents of `out/` to a static host at a domain root. Configure the
host to serve directory `index.html` files and `404.html` for missing routes.
No production deployment is configured. Hosting under a path prefix requires
setting Next.js `basePath` and adapting search/Markdown URLs, then rebuilding
and testing; the current build assumes `/`.

## Content conventions

- Describe the Python device and CUDA shim separately; their ports and protocols differ.
- Check commands and defaults against the implementation before changing docs.
- Document supported functionality; keep engineering proposals separate.
- Keep measurement scope and limitations beside benchmark numbers. Do not relabel
  historical results as fresh measurements or native baselines.
- Link to engineering documents for detailed evidence; avoid copying entire internal specs.

Sources: repository `README.md`, `docs/performance-notes.md`, the current Python
and C++ implementation. This site uses the official
[Fumadocs static Next.js template](https://www.fumadocs.dev/docs).
