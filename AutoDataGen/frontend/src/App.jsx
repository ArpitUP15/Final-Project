import React, { useState, useEffect } from 'react';
import { 
  Database, Cpu, ShieldCheck, TrendingUp, Users, 
  ChevronDown, Menu, X, ArrowRight, Sparkles, CheckCircle2, 
  RefreshCw, MessageSquare, LogOut, User, Lock, Mail, ArrowLeft
} from 'lucide-react';

export default function App() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [activeFaq, setActiveFaq] = useState(null);
  const [activeTab, setActiveTab] = useState('tabular');
  const [themeMode, setThemeMode] = useState('midnight');

  // Authentication states
  const [currentView, setCurrentView] = useState('home'); // 'home' or 'auth'
  const [authMode, setAuthMode] = useState('login'); // 'login' or 'signup'
  const [user, setUser] = useState(null);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');

  // Smooth scroll handler
  const scrollToSection = (id) => {
    if (currentView !== 'home') {
      setCurrentView('home');
      setTimeout(() => {
        const element = document.getElementById(id);
        if (element) element.scrollIntoView({ behavior: 'smooth' });
      }, 100);
      return;
    }
    setMobileMenuOpen(false);
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleAuthSubmit = (e) => {
    e.preventDefault();
    if (!email || !password || (authMode === 'signup' && !name)) {
      alert("Please fill in all fields.");
      return;
    }
    // Mock authentication success
    const loggedUser = {
      name: authMode === 'signup' ? name : email.split('@')[0],
      email: email
    };
    setUser(loggedUser);
    setCurrentView('home');
    setPassword('');
  };

  const handleLogout = () => {
    setUser(null);
    setCurrentView('home');
  };

  const faqs = [
    {
      q: "What is a Multi-Agent Dataset Augmentation system?",
      a: "AutotDataGen employs multiple autonomous agents that collaborate to analyze, augment, and improve accuracy of your training datasets."
    }, 
    {
      q: "How does AutotDataGen improve model accuracy?",
      a: "By filling sparse data gaps and introducing edge-case variations through multi-agent critique loops, downstream machine learning models achieve up to 35% higher robustness and generalization."
    },
    {
      q: "Can I export augmented data to custom formats?",
      a: "Yes! You can seamlessly export your datasets in JSON, CSV format."
    },
    {
      q: "Is my training data kept secure and private?",
      a: "All data transformations occur within isolated sandbox environments with end-to-end encryption, ensuring your proprietary data never leaks."
    }
  ];

  const team = [
    { name: "Arpit Bansal", role: "Project Lead & Backend Developer" },
    { name: "Aditya Saini", role: "Multi-Agent Systems Developer" },
    { name: "Aman Rizvi", role: "Multi-Agent Systems Developer" },
    { name: "Prachi Singh", role: "Frontend Developer" }
  ];

  const feedbacks = [
    {
      name: "Dr. R. K. Malhotra",
      role: "Head of AI Research, TechCorp",
      comment: "AutotDataGen solved our computer vision data scarcity issue in days instead of months. The multi-agent feedback loops are phenomenal."
    },
    {
      name: "Priya Deshmukh",
      role: "Lead Data Scientist",
      comment: "The ability to automatically generate realistic edge cases boosted our fraud detection model's accuracy by 28%. Absolute game-changer!"
    }
  ];

  const themeStyles = {
    midnight: {
      background: 'black',
      primary: 'from-cyan-500 via-indigo-500 to-violet-600',
      secondary: 'border-cyan-400/20 text-cyan-200 bg-cyan-500/10',
      navBorder: 'border-cyan-500/10',
      glow: 'shadow-cyan-500/25'
    }
  };

  const activeTheme = themeStyles[themeMode];

  if (currentView === 'auth') {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100 font-serif flex flex-col justify-center items-center px-6 relative overflow-hidden">
        {/* Glow Effects */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-125 h-125 bg-indigo-600/15 rounded-full blur-[120px] pointer-events-none" />

        <div className="w-full max-w-md bg-slate-900 border border-slate-800 rounded-3xl p-8 shadow-2xl relative z-10">
          <button 
            onClick={() => setCurrentView('home')}
            className="flex items-center gap-2 text-xs text-slate-400 hover:text-white mb-6 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" /> Back to Home
          </button>

          <div className="text-center mb-8">
            <div className="w-12 h-12 rounded-2xl bg-linear-to-tr from-indigo-600 to-violet-500 mx-auto flex items-center justify-center shadow-lg shadow-indigo-500/20 mb-3">
              <Database className="w-6 h-6 text-white" />
            </div>
            <h2 className="text-2xl font-bold text-white">
              {authMode === 'login' ? 'Welcome Back' : 'Create Account'}
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              {authMode === 'login' ? 'Sign in to access your data generation pipelines' : 'Start synthesizing high-accuracy datasets today'}
            </p>
          </div>

          <form onSubmit={handleAuthSubmit} className="space-y-4">
            {authMode === 'signup' && (
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1.5">Full Name</label>
                <div className="relative">
                  <User className="absolute left-3.5 top-3.5 w-4 h-4 text-slate-500" />
                  <input 
                    type="text" 
                    placeholder="John Doe" 
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-10 py-3 text-sm text-white focus:outline-none focus:border-indigo-500 transition-colors"
                  />
                </div>
              </div>
            )}

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Email Address</label>
              <div className="relative">
                <Mail className="absolute left-3.5 top-3.5 w-4 h-4 text-slate-500" />
                <input 
                  type="email" 
                  placeholder="name@company.com" 
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-10 py-3 text-sm text-white focus:outline-none focus:border-indigo-500 transition-colors"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Password</label>
              <div className="relative">
                <Lock className="absolute left-3.5 top-3.5 w-4 h-4 text-slate-500" />
                <input 
                  type="password" 
                  placeholder="••••••••" 
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-10 py-3 text-sm text-white focus:outline-none focus:border-indigo-500 transition-colors"
                />
              </div>
            </div>

            <button 
              type="submit"
              className="w-full mt-2 py-3.5 rounded-xl font-semibold bg-linear-to-r from-indigo-600 to-violet-600 text-white shadow-lg shadow-indigo-600/25 hover:opacity-90 transition-opacity"
            >
              {authMode === 'login' ? 'Sign In' : 'Create Account'}
            </button>
          </form>

          <div className="mt-6 text-center text-xs text-slate-400">
            {authMode === 'login' ? (
              <p>
                Don't have an account?{' '}
                <button onClick={() => setAuthMode('signup')} className="text-indigo-400 font-semibold hover:underline">
                  Sign up
                </button>
              </p>
            ) : (
              <p>
                Already have an account?{' '}
                <button onClick={() => setAuthMode('login')} className="text-indigo-400 font-semibold hover:underline">
                  Sign in
                </button>
              </p>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="min-h-screen text-slate-100 font-serif selection:bg-cyan-400 selection:text-slate-950 overflow-x-hidden"
      style={{ background: activeTheme.background }}
    >
      
      {/* 1. NAVIGATION BAR */}
      <nav className={`fixed top-0 left-0 right-0 z-50 bg-slate-950/75 backdrop-blur-xl border-b ${activeTheme.navBorder}`}>
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => scrollToSection('hero')}>
            <div className="w-10 h-10 rounded-xl bg-linear-to-tr to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/20 text-sm font-bold text-white">
              ADG
            </div>
            <span className="text-xl font-bold tracking-tight bg-linear-to-r from-white via-slate-200 to-indigo-400 bg-clip-text text-transparent">
              AutoDataGen
            </span>
          </div>

          {/* Desktop Links */}
          <div className="hidden md:flex items-center space-x-8 text-sm font-medium text-slate-300">
            <button onClick={() => scrollToSection('features')} className="hover:text-indigo-400 transition-colors">Features</button>
            <button onClick={() => scrollToSection('feedbacks')} className="hover:text-indigo-400 transition-colors">Feedbacks</button>
            <button onClick={() => scrollToSection('faqs')} className="hover:text-indigo-400 transition-colors">FAQs</button>
            <button onClick={() => scrollToSection('team')} className="hover:text-indigo-400 transition-colors">Our Team</button>
          </div>

          <div className="hidden md:flex items-center space-x-4">
            {user ? (
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-full bg-indigo-600/30 text-indigo-300 flex items-center justify-center text-xs font-bold border border-slate-700">
                  {user.name.charAt(0).toUpperCase()}
                </div>
                <button 
                  onClick={handleLogout}
                  className="p-2 rounded-xl bg-slate-900 border border-slate-800 text-slate-400 hover:text-red-400 hover:border-red-500/30 transition-all"
                  title="Logout"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <button 
                onClick={() => { setAuthMode('login'); setCurrentView('auth'); }}
                className={`relative group px-5 py-2.5 rounded-xl font-medium text-sm overflow-hidden bg-linear-to-r ${activeTheme.primary} text-white shadow-lg ${activeTheme.glow} transition-all duration-300 hover:scale-105 active:scale-95`}
              >
                <span className="relative z-10 flex items-center gap-2">
                  Get Started <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
                </span>
              </button>
            )}
          </div>

          {/* Mobile Menu Button */}
          <button className="md:hidden text-slate-300 hover:text-white" onClick={() => setMobileMenuOpen(!mobileMenuOpen)}>
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {/* Mobile Dropdown */}
        {mobileMenuOpen && (
          <div className="md:hidden absolute top-20 left-0 w-full bg-slate-900 border-b border-slate-800 p-6 flex flex-col space-y-4 shadow-2xl animate-in slide-in-from-top duration-200">
            <button onClick={() => scrollToSection('features')} className="text-left text-slate-300 hover:text-indigo-400">Features</button>
            <button onClick={() => scrollToSection('feedbacks')} className="text-left text-slate-300 hover:text-indigo-400">Feedbacks</button>
            <button onClick={() => scrollToSection('faqs')} className="text-left text-slate-300 hover:text-indigo-400">FAQs</button>
            <button onClick={() => scrollToSection('team')} className="text-left text-slate-300 hover:text-indigo-400">Our Team</button>
            {user ? (
              <div className="pt-4 border-t border-slate-800 flex items-center justify-between">
                <div className="w-8 h-8 rounded-full bg-indigo-600/30 text-indigo-300 flex items-center justify-center text-xs font-bold border border-slate-700">
                  {user.name.charAt(0).toUpperCase()}
                </div>
                <button onClick={handleLogout} className="text-red-400 text-sm font-medium flex items-center gap-1">
                  <LogOut className="w-4 h-4" /> Logout
                </button>
              </div>
            ) : (
              <button 
                onClick={() => { setAuthMode('login'); setCurrentView('auth'); }}
                className="w-full py-3 rounded-xl bg-indigo-600 text-white font-medium text-center shadow-lg"
              >
                Get Started
              </button>
            )}
          </div>
        )}
      </nav>

      {}
      <section id="hero" className="relative pt-32 pb-20 md:pt-44 md:pb-32 px-6 flex items-center justify-center overflow-hidden">
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-150 h-150 bg-indigo-600/15 rounded-full blur-[120px] pointer-events-none" />
        <div className="absolute top-1/3 right-10 w-100 h-100 bg-violet-600/10 rounded-full blur-[100px] pointer-events-none" />

        <div className="max-w-5xl mx-auto text-center relative z-10">
          <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full border text-xs md:text-sm font-medium mb-6 animate-bounce shadow-[0_0_25px_rgba(34,211,238,0.18)] ${activeTheme.secondary}`}>
            <Sparkles className="w-4 h-4" />
            <span>Next-Gen Multi-Agent Dataset Augmentation</span>
          </div>

          <h1 className="text-4xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight text-white mb-6 leading-tight">
            Scale M.L Models With <br />
            <span className="bg-linear-to-r from-indigo-400 via-violet-400 to-fuchsia-400 bg-clip-text text-transparent">
              Autonomous AI Agents
            </span>
          </h1>

          <p className="text-base sm:text-lg lg:text-xl text-slate-400 max-w-2xl mx-auto mb-10">
            AutoDataGen orchestrates cooperative multi-agent networks to synthesize, validate, and expand training data—supercharging model accuracy and overcoming data scarcity.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button 
              onClick={() => {
                if (user) {
                  window.location.href = 'http://127.0.0.1:5173/';
                } else {
                  setAuthMode('login');
                  setCurrentView('auth');
                }
              }}
              className={`w-full sm:w-auto px-8 py-4 rounded-xl font-semibold bg-linear-to-r ${activeTheme.primary} text-white shadow-xl ${activeTheme.glow} hover:scale-105 transition-all duration-300 flex items-center justify-center gap-2 group`}
            >
              Start Generating Data 
              <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
            </button>
            <button 
              onClick={() => scrollToSection('features')}
              className="w-full sm:w-auto px-8 py-4 rounded-xl font-semibold bg-slate-900/80 border border-slate-800 text-slate-300 hover:bg-slate-800 hover:text-white transition-all duration-300"
            >
              Explore Features
            </button>
          </div>

          <div className="mt-16 p-2 rounded-2xl bg-linear-to-b from-slate-800/80 to-slate-900/80 border border-slate-700/50 shadow-2xl relative group">
            <div className="absolute inset-0 bg-indigo-500/5 rounded-2xl filter blur-xl group-hover:bg-indigo-500/10 transition-all"></div>
            <div className="relative rounded-xl bg-slate-950 p-6 flex flex-col md:flex-row items-center justify-between gap-6 border border-slate-800">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
                  <Cpu className="w-6 h-6 text-indigo-400 animate-spin" style={{ animationDuration: '8s' }} />
                </div>
                <div className="text-left">
                  <h4 className="font-semibold text-white">Active Multi-Agent Pipeline</h4>
                  <p className="text-xs text-slate-400">Analysis Agent ➔ Strategy Selection Agent ➔ Execution Agent ➔ Evaluation Agent ➔ Orchestrator Agent</p>
                </div>
              </div>

            </div>
          </div>
        </div>
      </section>

      {}
      <section id="features" className="py-24 px-6 relative" style={{ background: 'linear-gradient(180deg, rgba(15,23,42,0.25), rgba(9,15,29,0.75))' }}>
        <div className="max-w-7xl mx-auto">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight text-white mb-4">
              Engineered for Ultimate Data Precision
            </h2>
            <p className="text-slate-400 text-sm md:text-base">
              Discover how AutoDataGen transforms raw inputs into robust, high-performing datasets through advanced automation.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="p-8 rounded-2xl bg-slate-900 border border-slate-800/80 hover:border-indigo-500/50 transition-all duration-300 group hover:-translate-y-1">
              <div className="w-14 h-14 rounded-xl bg-indigo-600/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 mb-6 group-hover:scale-110 transition-transform">
                <Sparkles className="w-7 h-7" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Generate Augmented Data</h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                Automatically synthesize domain-specific rows, image perturbations, or text variations matching statistical distributions of your seed data.
              </p>
            </div>

            <div className="p-8 rounded-2xl bg-slate-900 border border-slate-800/80 hover:border-indigo-500/50 transition-all duration-300 group hover:-translate-y-1">
              <div className="w-14 h-14 rounded-xl bg-emerald-600/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 mb-6 group-hover:scale-110 transition-transform">
                <TrendingUp className="w-7 h-7" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Improve Model Accuracy</h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                Mitigate overfitting and data scarcity. Multi-agent evaluation loops ensure synthesized entries directly contribute to higher test scores.
              </p>
            </div>

            <div className="p-8 rounded-2xl bg-slate-900 border border-slate-800/80 hover:border-indigo-500/50 transition-all duration-300 group hover:-translate-y-1">
              <div className="w-14 h-14 rounded-xl bg-violet-600/10 border border-violet-500/20 flex items-center justify-center text-violet-400 mb-6 group-hover:scale-110 transition-transform">
                <Cpu className="w-7 h-7" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Multi-Agent Consensus</h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                Autonomous agents cross-examine generated records for validity, factual consistency, and anomaly detection prior to export.
              </p>
            </div>

            <div className="p-8 rounded-2xl bg-slate-900 border border-slate-800/80 hover:border-indigo-500/50 transition-all duration-300 group hover:-translate-y-1">
              <div className="w-14 h-14 rounded-xl bg-blue-600/10 border border-blue-500/20 flex items-center justify-center text-blue-400 mb-6 group-hover:scale-110 transition-transform">
                <ShieldCheck className="w-7 h-7" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Bias & Privacy Filters</h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                Built-in sanitization agents detect and eliminate demographic bias and sensitive personally identifiable information (PII).
              </p>
            </div>

            <div className="p-8 rounded-2xl bg-slate-900 border border-slate-800/80 hover:border-indigo-500/50 transition-all duration-300 group hover:-translate-y-1">
              <div className="w-14 h-14 rounded-xl bg-amber-600/10 border border-amber-500/20 flex items-center justify-center text-amber-400 mb-6 group-hover:scale-110 transition-transform">
                <Database className="w-7 h-7" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Instant Format Export</h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                Export clean datasets instantly in CSV, JSON, Parquet, or stream directly into training pipelines with single-click API integration.
              </p>
            </div>

            <div className="p-8 rounded-2xl bg-slate-900 border border-slate-800/80 hover:border-indigo-500/50 transition-all duration-300 group hover:-translate-y-1">
              <div className="w-14 h-14 rounded-xl bg-fuchsia-600/10 border border-fuchsia-500/20 flex items-center justify-center text-fuchsia-400 mb-6 group-hover:scale-110 transition-transform">
                <RefreshCw className="w-7 h-7" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Iterative Fine-Tuning</h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                Continuously adapt generation prompts based on live validation loss metrics reported back from your model evaluation runs.
              </p>
            </div>
          </div>
        </div>
      </section>

      {}
      <section id="feedbacks" className="py-24 px-6 bg-slate-900/40 border-t border-b border-slate-800/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center max-w-2xl mx-auto mb-16">
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight text-white mb-4">
              Trusted by Leading Data Scientists
            </h2>
            <p className="text-slate-400 text-sm md:text-base">
              See what professionals say about their accuracy gains using AutotDataGen.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {feedbacks.map((fb, idx) => (
              <div key={idx} className="p-8 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col justify-between">
                <div className="mb-6">
                  <MessageSquare className="w-8 h-8 text-indigo-500 mb-4 opacity-80" />
                  <p className="text-slate-300 italic text-sm md:text-base leading-relaxed">
                    "{fb.comment}"
                  </p>
                </div>
                <div className="flex items-center gap-4 pt-4 border-t border-slate-800/80">
                  <div className="w-10 h-10 rounded-full bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center font-bold text-indigo-400 text-sm">
                    {fb.name.charAt(0)}
                  </div>
                  <div>
                    <h4 className="font-semibold text-white text-sm">{fb.name}</h4>
                    <span className="text-xs text-slate-400">{fb.role}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="faqs" className="py-24 px-6">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight text-white mb-4">
              Frequently Asked Questions
            </h2>
            <p className="text-slate-400">Everything you need to know about the AutotDataGen platform.</p>
          </div>

          <div className="space-y-4">
            {faqs.map((faq, index) => (
              <div 
                key={index}
                className="rounded-2xl bg-slate-900 border border-slate-800 overflow-hidden transition-all duration-200"
              >
                <button
                  onClick={() => setActiveFaq(activeFaq === index ? null : index)}
                  className="w-full p-6 text-left flex items-center justify-between font-semibold text-white focus:outline-none"
                >
                  <span className="text-base md:text-lg">{faq.q}</span>
                  <ChevronDown className={`w-5 h-5 text-indigo-400 transition-transform duration-300 ${activeFaq === index ? 'rotate-180' : ''}`} />
                </button>
                {activeFaq === index && (
                  <div className="px-6 pb-6 text-slate-400 text-sm leading-relaxed border-t border-slate-800/50 pt-4 animate-in fade-in duration-200">
                    {faq.a}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {}
      <section id="team" className="py-24 px-6 bg-slate-900/40 border-t border-slate-800/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center max-w-2xl mx-auto mb-16">
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight text-white mb-4">
              Meet Our Team
            </h2>
            <p className="text-slate-400">The minds behind AutotDataGen's success.</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-8">
            {team.map((member, index) => (
              <div
                key={index}
                className="rounded-2xl bg-slate-900 border border-slate-800 overflow-hidden group text-center p-6 hover:border-indigo-500/50 transition-all"
              >
                <h3 className="text-lg font-bold text-white mb-1">{member.name}</h3>
                <p className="text-indigo-400 text-xs font-medium uppercase tracking-wider">
                  {member.role}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <footer className="bg-slate-950 border-t border-slate-800/80 pt-16 pb-12 px-6">
        <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-10 mb-12">
          <div className="md:col-span-2">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
                <Database className="w-4 h-4 text-white" />
              </div>
              <span className="text-lg font-bold text-white">AutotDataGen</span>
            </div>
            <p className="text-slate-400 text-sm max-w-sm mb-6">
              Empowering next-generation machine learning with cooperative multi-agent dataset augmentation and automated validation pipelines.
            </p>
          </div>

          <div>
            <h4 className="font-semibold text-white text-sm mb-4">Quick Links</h4>
            <ul className="space-y-2 text-sm text-slate-400">
              <li><button onClick={() => scrollToSection('features')} className="hover:text-indigo-400">Features</button></li>
              <li><button onClick={() => scrollToSection('feedbacks')} className="hover:text-indigo-400">Feedbacks</button></li>
              <li><button onClick={() => scrollToSection('faqs')} className="hover:text-indigo-400">FAQs</button></li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-white text-sm mb-4">Legal & Privacy</h4>
            <ul className="space-y-2 text-sm text-slate-400">
              <li><a href="#privacy" className="hover:text-indigo-400">Privacy Policy</a></li>
              <li><a href="#terms" className="hover:text-indigo-400">Terms of Service</a></li>
              <li><a href="#security" className="hover:text-indigo-400">Data Security</a></li>
            </ul>
          </div>
        </div>

        <div className="max-w-7xl mx-auto pt-8 border-t border-slate-900 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500">
          <p>© {new Date().getFullYear()} AutotDataGen. All rights reserved.</p>
          <p className="mt-4 sm:mt-0">Built with React.js & Tailwind CSS</p>
        </div>
      </footer>

    </div>
  );
}