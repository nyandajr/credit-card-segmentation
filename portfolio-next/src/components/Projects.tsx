'use client';

import Image from 'next/image';
import { projects } from '@/data/projects';

export function Projects() {
  return (
    <section id="projects" className="bg-slate-950/60">
      <div className="container">
        <div className="mb-10">
          <h2 className="text-3xl font-bold text-white">Featured Projects</h2>
          <p className="mt-2 max-w-2xl text-slate-200">
            Real-world data science and ML systems that include reproducible code, automated pipelines, and business impact.
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          {projects.map((project) => (
            <article key={project.id} className="card flex flex-col overflow-hidden">
              {project.image ? (
                <div className="relative h-44 w-full overflow-hidden">
                  <Image
                    src={project.image}
                    alt={project.title}
                    fill
                    className="object-cover transition-transform duration-300 hover:scale-105"
                  />
                </div>
              ) : (
                <div className="flex h-44 items-center justify-center bg-white/5">
                  <span className="text-3xl">📁</span>
                </div>
              )}

              <div className="flex flex-1 flex-col gap-4 p-6">
                <div>
                  <h3 className="text-xl font-semibold text-white">{project.title}</h3>
                  <p className="mt-2 text-sm text-slate-200">{project.description}</p>
                </div>

                <div className="mt-auto flex flex-wrap gap-2">
                  {project.tech.map((tech) => (
                    <span
                      key={tech}
                      className="rounded-full bg-white/10 px-3 py-1 text-xs font-medium text-slate-200"
                    >
                      {tech}
                    </span>
                  ))}
                </div>

                <div className="mt-4 flex flex-wrap gap-2">
                  {project.repoUrl && (
                    <a
                      href={project.repoUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center justify-center rounded-full bg-brand-500 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-brand-400"
                    >
                      View code
                    </a>
                  )}
                  {project.liveUrl && (
                    <a
                      href={project.liveUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center justify-center rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-white/90 transition hover:bg-white/10"
                    >
                      Live demo
                    </a>
                  )}
                  {project.notebookUrl && (
                    <a
                      href={project.notebookUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center justify-center rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-white/90 transition hover:bg-white/10"
                    >
                      Notebook
                    </a>
                  )}
                </div>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
