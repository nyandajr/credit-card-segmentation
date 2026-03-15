export type ExperienceItem = {
  id: string;
  title: string;
  company: string;
  period: string;
  bullets: string[];
  type?: 'full-time' | 'contract' | 'internship' | 'research';
};

export const experience: ExperienceItem[] = [
  {
    id: 'mixx',
    title: 'Pricing & CVM Analyst',
    company: 'Mixx by Yas',
    period: 'Jan 2025 – Present',
    type: 'full-time',
    bullets: [
      'Develop predictive churn and revenue forecasting models using Python and SQL to support customer value management.',
      'Built behavioral customer segmentation models to optimize campaign targeting, improve ARPU, and enhance retention.',
      'Automated analytics pipelines to improve campaign execution speed and accuracy.',
      'Collaborated with commercial teams to translate business problems into data-driven AI solutions.',
    ],
  },
  {
    id: 'ai4d',
    title: 'Lead Data Scientist',
    company: 'AI4D Research Lab',
    period: 'Jan 2023 – Dec 2024',
    type: 'research',
    bullets: [
      'Led end-to-end machine learning lifecycle, from data acquisition and EDA to model development, deployment, and monitoring.',
      'Deployed deep learning-based prediction models using TensorFlow/PyTorch for early-risk identification.',
      'Improved model accuracy and stability through systematic feature engineering and tuning.',
      'Worked with multidisciplinary teams to deliver AI-driven insights for operational and policy planning.',
    ],
  },
];
