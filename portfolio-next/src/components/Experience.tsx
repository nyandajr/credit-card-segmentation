'use client';

import { experience } from '@/data/experience';

export function Experience() {
  return (
    <section id="experience" className="bg-slate-950/60">
      <div className="container">
        <div className="mb-10">
          <h2 className="text-3xl font-bold text-white">Work Experience</h2>
          <p className="mt-2 max-w-2xl text-slate-200">
            The roles below reflect where I’ve delivered product- and insight-driven AI systems. Each engagement is
            focused on turning data into measurable outcomes.
          </p>
        </div>
        <div className="space-y-10">
          {experience.map((item) => (
            <div key={item.id} className="grid gap-6 lg:grid-cols-12">
              <div className="lg:col-span-3">
                <p className="text-sm font-semibold uppercase tracking-wide text-brand-200">{item.period}</p>
                <h3 className="mt-3 text-xl font-semibold text-white">{item.title}</h3>
                <p className="text-sm text-slate-300">{item.company}</p>
              </div>

              <div className="lg:col-span-9">
                <ul className="space-y-3 text-slate-200">
                  {item.bullets.map((bullet) => (
                    <li key={bullet} className="flex gap-3">
                      <span className="mt-1 h-2 w-2 rounded-full bg-brand-300" />
                      <span>{bullet}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
