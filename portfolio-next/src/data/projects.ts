export type Project = {
  id: string;
  title: string;
  description: string;
  tech: string[];
  repoUrl?: string;
  liveUrl?: string;
  notebookUrl?: string;
  image?: string;
};

export const projects: Project[] = [
  {
    id: 'ea-financial-tracker',
    title: 'EA Financial Tracker',
    description:
      'Personal finance tracker with expense categorization, budget dashboards, and monthly summaries. Built as a Flask web app with interactive Plotly charts and user authentication.',
    tech: ['Flask', 'Plotly', 'SQLite', 'Bootstrap'],
    repoUrl: 'https://github.com/nyandajr/ea-financial-tracker',
    liveUrl: 'https://nyandajr.github.io/ea-financial-tracker',
    image: '/images/project-1.jpg',
  },
  {
    id: 'credit-card-segmentation',
    title: 'Credit Card Segmentation',
    description:
      'Developed customer segmentation using clustering and dimensionality reduction to identify high-value and at-risk cohorts. Includes interactive dashboards for segment profiling and targeting.',
    tech: ['Python', 'Scikit-learn', 'K-Means', 'Pandas'],
    repoUrl: 'https://github.com/nyandajr/credit-card-segmentation',
    notebookUrl: 'https://github.com/nyandajr/credit-card-segmentation/tree/main/notebooks',
    image: '/images/project-2.jpg',
  },
  {
    id: 'afya-check',
    title: 'Afya Check (Healthcare Triage App)',
    description:
      'A Flask-based symptom checker web app that guides users through triage questions and provides care recommendations. Built to support basic screening workflows in low-resource healthcare settings.',
    tech: ['Flask', 'Python', 'HTML/CSS', 'SQLite'],
    repoUrl: 'https://github.com/nyandajr/Afya_Check/tree/flask_changes',
    image: '/images/project-5.jpg',
  },
  {
    id: 'churn-prediction',
    title: 'Customer Churn Prediction',
    description:
      'End-to-end churn prediction pipeline using Logistic Regression, Random Forest, and XGBoost on transaction data. Includes model monitoring dashboards and feature importance analysis.',
    tech: ['Python', 'XGBoost', 'Scikit-learn', 'Dash'],
    repoUrl: 'https://github.com/nyandajr/churn-prediction',
    notebookUrl: 'https://github.com/nyandajr/churn-prediction/blob/main/notebooks/churn_analysis.ipynb',
    image: '/images/project-6.jpg',
  },
];
