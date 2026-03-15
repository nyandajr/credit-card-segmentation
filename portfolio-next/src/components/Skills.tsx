'use client';

const skillCategories = [
  {
    title: 'ML & Analytics',
    icon: '🧠',
    skills: ['Predictive Modeling', 'Customer Segmentation', 'Time Series Forecasting', 'Feature Engineering', 'Model Optimization', 'Statistical Analysis', 'EDA', 'Hypothesis Testing'],
  },
  {
    title: 'Data Platforms',
    icon: '🗄️',
    skills: ['SQL Server', 'PySpark', 'Data Lakes', 'ETL Pipelines', 'Power BI', 'Tableau', 'REST APIs'],
  },
  {
    title: 'Engineering',
    icon: '⚙️',
    skills: ['Python', 'Docker', 'Airflow', 'API Development', 'CI/CD', 'Unit Testing'],
  },
];

export function Skills() {
  return (
    <section id="skills">
      <div className="container">
        <div className="mb-10">
          <h2 className="text-3xl font-bold text-white">Skills & Competencies</h2>
          <p className="mt-2 max-w-2xl text-slate-200">
            A snapshot of the tools and techniques I use to design, build, and deploy data products.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          {skillCategories.map((category) => (
            <div key={category.title} className="card">
              <h3 className="flex items-center gap-3 text-xl font-semibold text-white">
                <span>{category.icon}</span>
                {category.title}
              </h3>
              <div className="mt-4 flex flex-wrap gap-2">
                {category.skills.map((skill) => (
                  <span
                    key={skill}
                    className="rounded-full bg-white/10 px-4 py-2 text-xs font-medium text-slate-100"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
