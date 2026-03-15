'use client';

import Image from 'next/image';

const stats = [
  { label: 'Models deployed', value: '5+' },
  { label: 'Data processed', value: '200M+' },
  { label: 'Projects shipped', value: '12+' },
];

export function Hero() {
  return (
    <section id="home" className="relative overflow-hidden bg-gradient-to-b from-slate-950 via-slate-950/90 to-slate-950">
      <div className="container grid gap-12 md:grid-cols-2 items-center py-24">
        <div className="space-y-6">
          <p className="inline-flex items-center gap-2 rounded-full bg-white/5 px-4 py-2 text-xs font-semibold uppercase tracking-wide text-white/80">
            Data Scientist • AI Engineer
          </p>
          <h1 className="text-4xl font-extrabold tracking-tight text-white sm:text-5xl">
            Freddy <span className="text-brand-300">Nyanda</span>
          </h1>
          <p className="max-w-xl leading-relaxed text-slate-200">
            I build predictive ML systems and data products that help teams increase retention, grow revenue, and
            deliver actionable insights at scale.
          </p>

          <div className="flex flex-wrap gap-3">
            <a
              href="#projects"
              className="inline-flex items-center justify-center rounded-full bg-brand-500 px-6 py-3 text-sm font-semibold text-slate-950 shadow-lg shadow-brand-500/20 transition hover:bg-brand-400"
            >
              View projects
            </a>
            <a
              href="#contact"
              className="inline-flex items-center justify-center rounded-full border border-white/10 bg-white/5 px-6 py-3 text-sm font-semibold text-white/90 transition hover:bg-white/10"
            >
              Get in touch
            </a>
          </div>

          <div className="mt-10 flex flex-wrap gap-4">
            {stats.map((stat) => (
              <div key={stat.label} className="rounded-2xl border border-white/10 bg-white/5 px-5 py-4">
                <p className="text-2xl font-semibold text-white">{stat.value}</p>
                <p className="text-sm text-slate-300">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="relative flex items-center justify-center">
          <div className="absolute -top-14 -right-10 h-56 w-56 rounded-full bg-brand-500/20 blur-3xl" />
          <div className="relative rounded-3xl border border-white/10 bg-white/5 p-6 shadow-xl">
            <Image
              src="https://raw.githubusercontent.com/nyandajr/portfolio_site/main/images/freddy.png"
              alt="Freddy Nyanda"
              width={420}
              height={420}
              className="rounded-2xl object-cover"
              priority
            />
          </div>
        </div>
      </div>
    </section>
  );
}
