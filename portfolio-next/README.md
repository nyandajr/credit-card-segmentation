# Portfolio Next.js (Revamp)

This repository contains a modern Next.js (App Router) version of the portfolio website.

## Goals
- Convert existing HTML/CSS portfolio into a React/Next.js app
- Add a more modern data-science portfolio UI (projects with links, live metrics, etc.)
- Make it easy to update projects via data-driven JSON/TS content

## Getting started
> Requires Node.js (>=18) and npm.

```bash
cd portfolio-next
npm install
npm run dev
```

Then open http://localhost:3000

## Deploy
- **Vercel** (recommended): connect this repo and deploy.
- **GitHub Pages** (static export): `npm run build && npm run export` (output goes to `out/`).

## Notes
- Replace placeholder URLs (GitHub, live demos, email) in `src/data/projects.ts` and `src/components/Contact.tsx`.
- Update the profile image at `public/images/freddy.png` and any project images to match your branding.
