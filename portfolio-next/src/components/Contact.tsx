'use client';

export function Contact() {
  return (
    <section id="contact">
      <div className="container">
        <div className="mb-10 text-center">
          <h2 className="text-3xl font-bold text-white">Get in touch</h2>
          <p className="mt-2 max-w-2xl mx-auto text-slate-200">
            Interested in collaborating or have a project in mind? Send a message and I’ll get back to you as soon as possible.
          </p>
        </div>

        <div className="mx-auto max-w-2xl">
          <form
            action="mailto:freddy.nyanda@example.com"
            method="post"
            encType="text/plain"
            className="grid gap-4"
          >
            <div className="grid gap-4 md:grid-cols-2">
              <label className="block">
                <span className="text-sm font-medium text-slate-200">Name</span>
                <input
                  type="text"
                  name="name"
                  required
                  className="mt-2 w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-slate-100 outline-none transition focus:border-brand-300 focus:ring-2 focus:ring-brand-300/40"
                />
              </label>
              <label className="block">
                <span className="text-sm font-medium text-slate-200">Email</span>
                <input
                  type="email"
                  name="email"
                  required
                  className="mt-2 w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-slate-100 outline-none transition focus:border-brand-300 focus:ring-2 focus:ring-brand-300/40"
                />
              </label>
            </div>

            <label className="block">
              <span className="text-sm font-medium text-slate-200">Message</span>
              <textarea
                name="message"
                rows={5}
                required
                className="mt-2 w-full resize-none rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-slate-100 outline-none transition focus:border-brand-300 focus:ring-2 focus:ring-brand-300/40"
              />
            </label>

            <button
              type="submit"
              className="inline-flex w-full items-center justify-center rounded-full bg-brand-500 px-6 py-3 text-sm font-semibold text-slate-950 transition hover:bg-brand-400"
            >
              Send message
            </button>
          </form>

          <div className="mt-10 rounded-2xl border border-white/10 bg-white/5 p-6">
            <h3 className="text-lg font-semibold text-white">Need a quicker response?</h3>
            <p className="mt-2 text-sm text-slate-200">
              You can also reach me via email at{' '}
              <a
                href="mailto:freddy.nyanda@example.com"
                className="text-brand-200 hover:text-brand-100"
              >
                freddy.nyanda@example.com
              </a>
              .
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
