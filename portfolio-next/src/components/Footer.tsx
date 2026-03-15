export function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="border-t border-white/10 bg-slate-950/40">
      <div className="container flex flex-col gap-3 py-10 text-center text-sm text-slate-400">
        <p>
          Built with Next.js · <a href="https://github.com/nyandajr" className="text-brand-200 hover:text-brand-100">GitHub</a>
        </p>
        <p>© {year} Freddy Nyanda. All rights reserved.</p>
      </div>
    </footer>
  );
}
