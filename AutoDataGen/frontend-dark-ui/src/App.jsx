import { useState } from 'react';
import {
  ArrowDownRight, ArrowLeft, ArrowRight, BarChart3, Check, ChevronDown,
  Database, FileImage, FileText, Layers3, LockKeyhole, Mail, Menu,
  Network, ShieldCheck, Table2, UserRound, X
} from 'lucide-react';
import './App.css';

const agents = [
  { number: '01', name: 'Analysis', detail: 'Profiles data quality and class balance.' },
  { number: '02', name: 'Strategy', detail: 'Selects a suitable augmentation method.' },
  { number: '03', name: 'Execution', detail: 'Generates candidate samples.' },
  { number: '04', name: 'Evaluation', detail: 'Measures the effect on model performance.' },
  { number: '05', name: 'Orchestrator', detail: 'Coordinates each stage and iteration.' },
];

const faqs = [
  { q: 'What does AutoDataGen do?', a: 'AutoDataGen is a multi-agent workflow for analyzing a dataset, selecting an augmentation approach, generating candidate data, and evaluating the result.' },
  { q: 'Which dataset types can I work with?', a: 'The interface is designed around tabular, text, and image datasets. Available formats and limits depend on the connected backend.' },
  { q: 'Does augmentation guarantee higher model accuracy?', a: 'No. Results depend on the dataset, task, model, and evaluation method. AutoDataGen is designed to measure changes so you can review whether an augmentation helped.' },
  { q: 'How is my data handled?', a: 'Data handling depends on the deployment and backend configuration. Review your environment’s storage, retention, and access settings before uploading sensitive data.' },
];

const datasetTypes = [
  { key: 'tabular', label: 'Tabular', icon: Table2, hint: 'Rows and columns' },
  { key: 'text', label: 'Text', icon: FileText, hint: 'Documents and labels' },
  { key: 'image', label: 'Image', icon: FileImage, hint: 'Image collections' },
];

const workflowUrl = 'http://127.0.0.1:5173/';

export default function App() {
  const [currentView, setCurrentView] = useState('home');
  const [authMode, setAuthMode] = useState('login');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [activeFaq, setActiveFaq] = useState(0);
  const [activeType, setActiveType] = useState('tabular');
  const [user, setUser] = useState(null);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');

  const goTo = (id) => {
    setMobileMenuOpen(false);
    if (currentView !== 'home') setCurrentView('home');
    window.setTimeout(() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' }), 0);
  };

  const openAuth = (mode = 'login') => {
    setAuthMode(mode);
    setCurrentView('auth');
    setMobileMenuOpen(false);
  };

  const handleAuthSubmit = (event) => {
    event.preventDefault();
    if (!email.trim() || !password || (authMode === 'signup' && !name.trim())) return;
    setUser({ name: authMode === 'signup' ? name.trim() : email.split('@')[0], email });
    setPassword('');
    setCurrentView('home');
  };

  const startWorkflow = () => {
    if (user) {
      window.location.assign(workflowUrl);
      return;
    }
    openAuth('signup');
  };

  if (currentView === 'auth') {
    return (
      <main className="auth-screen">
        <div className="auth-card">
          <button className="text-button auth-back" onClick={() => setCurrentView('home')}>
            <ArrowLeft size={16} /> Back to home
          </button>
          <div className="brand-mark auth-mark"><Database size={22} /></div>
          <p className="eyebrow">AUTODATAGEN WORKSPACE</p>
          <h1>{authMode === 'login' ? 'Welcome back.' : 'Create your account.'}</h1>
          <p className="auth-intro">{authMode === 'login' ? 'Sign in to continue to your data workflow.' : 'Set up an account to start a dataset workflow.'}</p>
          <form onSubmit={handleAuthSubmit} className="auth-form">
            {authMode === 'signup' && <label>Name<span className="input-wrap"><UserRound size={17} /><input required value={name} onChange={(event) => setName(event.target.value)} placeholder="Your name" autoComplete="name" /></span></label>}
            <label>Email address<span className="input-wrap"><Mail size={17} /><input required type="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="you@example.com" autoComplete="email" /></span></label>
            <label>Password<span className="input-wrap"><LockKeyhole size={17} /><input required type="password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Enter your password" autoComplete={authMode === 'login' ? 'current-password' : 'new-password'} /></span></label>
            <button className="button button-primary auth-submit" type="submit">{authMode === 'login' ? 'Sign in' : 'Create account'}<ArrowRight size={17} /></button>
          </form>
          <p className="auth-switch">{authMode === 'login' ? 'New to AutoDataGen?' : 'Already have an account?'} <button className="text-button accent-text" onClick={() => setAuthMode(authMode === 'login' ? 'signup' : 'login')}>{authMode === 'login' ? 'Create an account' : 'Sign in'}</button></p>
          <p className="auth-note">Demo interface: connect your authentication service before using this form with real accounts.</p>
        </div>
      </main>
    );
  }

  return (
    <div className="site-shell">
      <header className="site-header">
        <a className="brand" href="#top" onClick={(event) => { event.preventDefault(); goTo('top'); }}>
          <span className="brand-mark"><Database size={19} /></span>
          <span>AutoDataGen</span>
        </a>
        <nav className={`main-nav ${mobileMenuOpen ? 'is-open' : ''}`} aria-label="Main navigation">
          <button onClick={() => goTo('workflow')}>Workflow</button>
          <button onClick={() => goTo('capabilities')}>Capabilities</button>
          <button onClick={() => goTo('team')}>Team</button>
          <button onClick={() => goTo('faq')}>FAQ</button>
          <div className="mobile-nav-action">{user ? <button onClick={() => { setUser(null); setMobileMenuOpen(false); }}>Sign out</button> : <button onClick={() => openAuth('login')}>Sign in</button>}</div>
        </nav>
        <div className="header-action">
          {user ? <><span className="signed-in"><span className="status-dot" />{user.name}</span><button className="button button-small button-outline" onClick={() => setUser(null)}>Sign out</button></> : <button className="button button-small button-outline" onClick={() => openAuth('login')}>Sign in <ArrowRight size={15} /></button>}
        </div>
        <button className="menu-toggle" onClick={() => setMobileMenuOpen(!mobileMenuOpen)} aria-label={mobileMenuOpen ? 'Close menu' : 'Open menu'}>{mobileMenuOpen ? <X /> : <Menu />}</button>
      </header>

      <main id="top">
        <section className="hero section-wrap">
          <div className="hero-copy">
            <p className="eyebrow"><span className="eyebrow-rule" /> MULTI-AGENT DATASET WORKFLOW</p>
            <h1>Better training data,<br /><span>built with intent.</span></h1>
            <p className="hero-description">Analyze a dataset, choose an augmentation strategy, and evaluate the result through a coordinated agent workflow.</p>
            <div className="hero-actions">
              <button className="button button-primary" onClick={startWorkflow}>Start Generating <ArrowRight size={17} /></button>
              <button className="button button-quiet" onClick={() => goTo('workflow')}>See how it works <ArrowDownRight size={17} /></button>
            </div>
            <div className="hero-footnote"><span className="status-dot" /> Five agents. One reviewable process.</div>
          </div>

          <div className="workflow-card" aria-label="AutoDataGen workflow preview">
            <div className="workflow-card-head"><div><span className="card-kicker">WORKFLOW PREVIEW</span><h2>Dataset augmentation</h2></div><span className="live-label"><span className="status-dot" /> READY</span></div>
            <div className="dataset-type" id="dataset-type">
              <p className="mini-label">CHOOSE YOUR DATA TYPE</p>
              <div className="type-options">{datasetTypes.map(({ key, label, icon: Icon, hint }) => <button key={key} className={`type-option ${activeType === key ? 'selected' : ''}`} onClick={() => setActiveType(key)}><Icon size={18} /><span>{label}</span><small>{hint}</small>{activeType === key && <Check className="type-check" size={15} />}</button>)}</div>
            </div>
            <div className="workflow-progress"><span>AGENT SEQUENCE</span><span>05 STAGES</span></div>
            <ol className="agent-list">{agents.map((agent, index) => <li className="agent-row" key={agent.number}><span className="agent-number">{agent.number}</span><span className="agent-node" aria-hidden="true">{index === 0 ? <BarChart3 size={15} /> : index === 1 ? <Layers3 size={15} /> : index === 2 ? <Database size={15} /> : index === 3 ? <ShieldCheck size={15} /> : <Network size={15} />}</span><span className="agent-name">{agent.name}<small>{agent.detail}</small></span><span className="agent-state">{index === 0 ? 'START' : 'NEXT'}</span></li>)}</ol>
            <div className="workflow-card-foot"><span>Selection: <strong>{datasetTypes.find((item) => item.key === activeType)?.label}</strong></span><span>Review each stage before export</span></div>
          </div>
        </section>

        <section className="workflow-section section-wrap" id="workflow">
          <div className="section-heading"><p className="eyebrow"><span className="eyebrow-rule" /> THE PROCESS</p><h2>From raw data to<br />measured results.</h2></div>
          <div className="process-grid">{[
            ['01', 'Inspect', 'Identify missing values, imbalance, duplicates, and dataset characteristics.'],
            ['02', 'Augment', 'Select a method that fits the data type and the issue being addressed.'],
            ['03', 'Evaluate', 'Compare outcomes with a consistent evaluation setup before accepting changes.'],
          ].map(([num, title, body]) => <article className="process-card" key={num}><span className="process-number">{num}</span><h3>{title}</h3><p>{body}</p><span className="process-line" /></article>)}</div>
        </section>

        <section className="capabilities-section" id="capabilities"><div className="section-wrap capabilities-inner"><div className="capabilities-heading"><p className="eyebrow"><span className="eyebrow-rule" /> CAPABILITIES</p><h2>Useful tools for<br />careful data work.</h2><p>Designed to make each dataset operation visible and each evaluation easier to review.</p></div><div className="capability-list">{[
          ['01', 'Dataset profiling', 'Summarize data quality and key distribution patterns.'],
          ['02', 'Method selection', 'Choose augmentation approaches based on the input data.'],
          ['03', 'Quality checks', 'Review generated samples for structure and consistency.'],
          ['04', 'Performance evaluation', 'Compare baseline and augmented data results.'],
          ['05', 'Export-ready output', 'Prepare processed data for downstream model workflows.'],
        ].map(([num, title, body]) => <article className="capability-row" key={num}><span>{num}</span><div><h3>{title}</h3><p>{body}</p></div><ArrowDownRight size={17} /></article>)}</div></div></section>

        <section className="team-section section-wrap" id="team"><div className="team-top"><div><p className="eyebrow"><span className="eyebrow-rule" /> PROJECT TEAM</p><h2>Built by the team.</h2></div><p>A student project exploring practical dataset augmentation through multi-agent coordination.</p></div><div className="team-grid">{[
          ['Arpit Bansal', 'Project Lead · Backend'], ['Aditya Saini', 'Multi-Agent Systems'], ['Aman Rizvi', 'Multi-Agent Systems'], ['Prachi Singh', 'Frontend'],
        ].map(([member, role], index) => <article className="team-card" key={member}><span className="team-index">0{index + 1}</span><span className="team-avatar">{member.split(' ').map((part) => part[0]).join('')}</span><h3>{member}</h3><p>{role}</p></article>)}</div></section>

        <section className="faq-section" id="faq"><div className="section-wrap faq-inner"><div><p className="eyebrow"><span className="eyebrow-rule" /> QUESTIONS</p><h2>Before you<br />get started.</h2><p className="faq-aside">Understand the workflow and its limits before running it on your data.</p></div><div className="faq-list">{faqs.map((faq, index) => <article className={`faq-item ${activeFaq === index ? 'active' : ''}`} key={faq.q}><button onClick={() => setActiveFaq(activeFaq === index ? null : index)} aria-expanded={activeFaq === index}><span className="faq-index">0{index + 1}</span><span>{faq.q}</span><ChevronDown size={18} /></button>{activeFaq === index && <p className="faq-answer">{faq.a}</p>}</article>)}</div></div></section>
      </main>

      <footer className="site-footer"><div className="section-wrap footer-main"><a className="brand" href="#top" onClick={(event) => { event.preventDefault(); goTo('top'); }}><span className="brand-mark"><Database size={18} /></span><span>AutoDataGen</span></a><p>Dataset augmentation, made inspectable.</p><button className="footer-top" onClick={() => goTo('top')}>Back to top <ArrowRight size={15} /></button></div><div className="section-wrap footer-bottom"><span>© {new Date().getFullYear()} AutoDataGen</span><span>Dark interface · Built with React</span></div></footer>
    </div>
  );
}
