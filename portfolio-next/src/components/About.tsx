'use client';

export function About() {
  return (
    <section id="about" className="bg-slate-950/60">
      <div className="container grid gap-10 lg:grid-cols-2 items-start">
        <div className="space-y-6">
          <h2 className="text-3xl font-bold text-white">About Me</h2>
          <p className="text-slate-200">
            I am a Data Scientist with <span className="font-semibold">4+ years of experience</span> developing
            predictive and prescriptive machine learning solutions across mobile money, telecom, and research
            environments.
          </p>
          <p className="text-slate-200">
            My expertise lies in building robust models for <span className="font-semibold">customer segmentation, churn
            prediction, and revenue forecasting</span>. I have hands-on experience deploying deep learning models into
            production and translating complex business challenges into actionable AI-driven insights.
          </p>
          <p className="text-slate-200">
            Currently, I work as a Pricing & CVM Analyst at Mixx by Yas, where I leverage data to optimize pricing strategies
            and enhance customer value.
          </p>

          <div className="grid gap-4 sm:grid-cols-3">
            <div className="card">
              <h3 className="text-sm font-semibold text-white">Experience</h3>
              <p className="mt-2 text-sm text-slate-200">4+ Years</p>
            </div>
            <div className="card">
              <h3 className="text-sm font-semibold text-white">Specialization</h3>
              <p className="mt-2 text-sm text-slate-200">Machine Learning</p>
            </div>
            <div className="card">
              <h3 className="text-sm font-semibold text-white">Location</h3>
              <p className="mt-2 text-sm text-slate-200">Dar es Salaam</p>
            </div>
          </div>
        </div>

        <div className="glass p-10">
          <h3 className="text-xl font-semibold text-white">What I enjoy building</h3>
          <ul className="mt-5 space-y-3 text-slate-200">
            <li>• End-to-end machine learning pipelines (data, model, deployment)</li>
            <li>• Customer analytics dashboards & segmentation tools</li>
            <li>• Forecasting and optimization models for revenue & churn</li>
            <li>• Production-grade APIs and data products</li>
          </ul>
        </div>
      </div>
    </section>
  );
}
