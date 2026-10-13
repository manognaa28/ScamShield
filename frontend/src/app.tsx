import { useEffect, useMemo, useState } from 'react'

const API_URL = 'http://127.0.0.1:8000'

interface JobAd {
  title: string
  company_profile: string
  description: string
  requirements: string
  benefits: string
}
interface PredictionResult {
  prediction: 'fraudulent' | 'legitimate'
  risk_score: number
  risk_level: 'High' | 'Moderate' | 'Low'
  disclaimer: string
  risk_level_note: string
}
interface HistoryItem extends JobAd {
  id: string
  prediction: 'fraudulent' | 'legitimate'
  risk_score: number
  risk_level: 'High' | 'Moderate' | 'Low'
  created_at: string
}
type Page = 'dashboard' | 'analyzer' | 'history'
type RiskFilter = 'All' | 'High' | 'Moderate' | 'Low'

const emptyForm: JobAd = { title: '', company_profile: '', description: '', requirements: '', benefits: '' }

function App() {
  const [form, setForm] = useState<JobAd>({ ...emptyForm })
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [page, setPage] = useState<Page>('dashboard')
  const [search, setSearch] = useState('')
  const [riskFilter, setRiskFilter] = useState<RiskFilter>('All')

  async function fetchHistory() {
    setHistoryLoading(true)
    setHistoryError('')
    try {
      const response = await fetch(`${API_URL}/history?limit=50`)
      if (!response.ok) throw new Error('Unable to load analysis history.')
      const data: HistoryItem[] = await response.json()
      setHistory(Array.isArray(data) ? data : [])
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : 'Something went wrong while loading history.')
    } finally {
      setHistoryLoading(false)
    }
  }

  useEffect(() => { void fetchHistory() }, [])

  function handleChange(event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) {
    const { name, value } = event.target
    setForm(previous => ({ ...previous, [name]: value }))
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')
    setResult(null)
    if (!Object.values(form).some(value => value.trim())) {
      setError('Please enter at least one job-ad field before analyzing.')
      return
    }
    setLoading(true)
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Please check the information and try again.')
      setResult(data as PredictionResult)
      await fetchHistory()
    } catch (err) {
      setError(err instanceof TypeError
        ? 'Cannot connect to ScamShield. Make sure the FastAPI backend is running on port 8000.'
        : err instanceof Error ? err.message : 'Something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  function loadAdvertisement(item: HistoryItem) {
    setForm({
      title: item.title || '', company_profile: item.company_profile || '',
      description: item.description || '', requirements: item.requirements || '', benefits: item.benefits || '',
    })
    setResult(null)
    setError('')
    setPage('analyzer')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const stats = useMemo(() => ({
    total: history.length,
    high: history.filter(item => item.risk_level === 'High').length,
    moderate: history.filter(item => item.risk_level === 'Moderate').length,
    low: history.filter(item => item.risk_level === 'Low').length,
    fraudulent: history.filter(item => item.prediction === 'fraudulent').length,
    legitimate: history.filter(item => item.prediction === 'legitimate').length,
  }), [history])

  const filteredHistory = useMemo(() => history.filter(item => {
    const q = search.trim().toLowerCase()
    const matchesRisk = riskFilter === 'All' || item.risk_level === riskFilter
    const matchesText = !q || [item.title, item.company_profile, item.description].some(text => (text || '').toLowerCase().includes(q))
    return matchesRisk && matchesText
  }), [history, search, riskFilter])

  return (
    <main className="scamshield-layout">
      <aside className="app-sidebar">
        <button className="sidebar-brand" type="button" onClick={() => setPage('dashboard')}><span className="sidebar-brand-icon">🛡️</span><span>Scam<span className="brand-accent">Shield</span></span></button>
        <p className="sidebar-label">WORKSPACE</p>
        <nav className="sidebar-nav" aria-label="Main navigation">
          <button className={page === 'dashboard' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('dashboard')} type="button">▦ &nbsp; Dashboard</button>
          <button className={page === 'analyzer' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('analyzer')} type="button">⌕ &nbsp; Job Analyzer</button>
          <button className={page === 'history' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('history')} type="button">◷ &nbsp; Analysis History <span className="nav-count">{history.length}</span></button>
        </nav>
        <div className="sidebar-bottom"><span className="sidebar-status-dot" /><div><strong>ML system</strong><span>Local API connection</span></div></div>
      </aside>

      <section className="app-main">
        <header className="topbar">
          <div><p className="eyebrow">SCAMSHIELD WORKSPACE</p><h1>{page === 'dashboard' ? 'Dashboard' : page === 'analyzer' ? 'Job Analyzer' : 'Analysis History'}</h1></div>
          <div className="topbar-status"><span className="status-dot" /> ML-powered analysis</div>
        </header>

        {page === 'dashboard' && <section className="page-content">
          <div className="welcome-panel">
            <div><p className="eyebrow">YOUR JOB SEARCH, WITH MORE CONFIDENCE</p><h2>Spot warning signs <span>before you apply.</span></h2><p>Review saved job-ad analyses and use machine-learning estimates as one signal when evaluating a posting.</p><button className="primary-button" type="button" onClick={() => setPage('analyzer')}>Analyze a job advertisement →</button></div>
            <div className="welcome-shield" aria-hidden="true">🛡️</div>
          </div>
          {historyError && <div className="error-message" role="alert">{historyError}</div>}
          <div className="stats-grid">
            <StatCard label="Total analyses" value={stats.total} caption="Saved advertisements" />
            <StatCard label="High risk" value={stats.high} caption="Review carefully" tone="high" />
            <StatCard label="Moderate risk" value={stats.moderate} caption="Check details" tone="moderate" />
            <StatCard label="Low risk" value={stats.low} caption="Not a guarantee" tone="low" />
          </div>
          <div className="dashboard-grid">
            <section className="panel-card"><div className="panel-heading"><div><p className="eyebrow">OVERVIEW</p><h2>Classification summary</h2></div><span className="subtle-label">Saved analyses</span></div>
              <div className="classification-row"><span>🔴 Potential warning signs</span><strong>{stats.fraudulent}</strong></div><div className="mini-track"><span className="fraudulent-bar" style={{ width: `${stats.total ? stats.fraudulent / stats.total * 100 : 0}%` }} /></div>
              <div className="classification-row"><span>🟢 Classified as legitimate</span><strong>{stats.legitimate}</strong></div><div className="mini-track"><span className="legitimate-bar" style={{ width: `${stats.total ? stats.legitimate / stats.total * 100 : 0}%` }} /></div>
              {stats.total === 0 && <p className="empty-note">Your summary will appear after you analyze a job advertisement.</p>}
            </section>
            <section className="panel-card"><div className="panel-heading"><div><p className="eyebrow">RISK BREAKDOWN</p><h2>Risk distribution</h2></div></div>
              {([['High', stats.high, 'risk-high'], ['Moderate', stats.moderate, 'risk-moderate'], ['Low', stats.low, 'risk-low']] as const).map(([label, value, css]) => <div className="risk-distribution-row" key={label}><div className="risk-distribution-top"><span>{label}</span><strong>{value}</strong></div><div className="mini-track"><span className={css} style={{ width: `${stats.total ? value / stats.total * 100 : 0}%` }} /></div></div>)}
            </section>
          </div>
          <section className="panel-card"><div className="panel-heading"><div><p className="eyebrow">LATEST ACTIVITY</p><h2>Recent analyses</h2></div><button className="text-button" type="button" onClick={() => setPage('history')}>View all →</button></div>
            {historyLoading ? <p className="empty-note">Loading saved analyses...</p> : history.length === 0 ? <div className="empty-state"><span className="empty-state-icon">⌕</span><h3>No analyses yet</h3><p>Analyze a job advertisement to start building your dashboard.</p><button className="secondary-button" type="button" onClick={() => setPage('analyzer')}>Open Job Analyzer</button></div> : <div className="recent-list">{history.slice(0, 5).map(item => <HistoryRow key={item.id} item={item} onLoad={() => loadAdvertisement(item)} />)}</div>}
          </section>
          <p className="dashboard-disclaimer">Scores are model-generated estimates, not proof that a job is fraudulent or genuine.</p>
        </section>}

        {page === 'analyzer' && <section className="page-content">
          <div className="page-intro"><p>Enter the information available from a job posting. You do not need to fill every field.</p></div>
          <div className="analyzer-grid">
            <form className="job-form panel-card" onSubmit={handleSubmit}>
              <div className="panel-heading"><div><p className="eyebrow">JOB DETAILS</p><h2>Job advertisement</h2></div><span className="subtle-label">01 / ANALYZE</span></div>
              <div className="form-field"><label htmlFor="title">Job title</label><input id="title" name="title" value={form.title} onChange={handleChange} maxLength={300} placeholder="e.g. Junior Software Developer" /></div>
              <div className="form-field"><label htmlFor="company_profile">Company profile</label><textarea id="company_profile" name="company_profile" value={form.company_profile} onChange={handleChange} maxLength={20000} rows={3} placeholder="Paste information about the company..." /></div>
              <div className="form-field"><label htmlFor="description">Job description</label><textarea id="description" name="description" value={form.description} onChange={handleChange} maxLength={50000} rows={5} placeholder="Paste the job advertisement here..." /></div>
              <div className="form-field"><label htmlFor="requirements">Requirements</label><textarea id="requirements" name="requirements" value={form.requirements} onChange={handleChange} maxLength={20000} rows={3} placeholder="Skills, qualifications and experience..." /></div>
              <div className="form-field"><label htmlFor="benefits">Benefits</label><textarea id="benefits" name="benefits" value={form.benefits} onChange={handleChange} maxLength={20000} rows={3} placeholder="Salary, benefits and other offers..." /></div>
              {error && <div className="error-message" role="alert"><strong>Unable to analyze</strong><p>{error}</p></div>}
              <div className="form-actions"><button className="primary-button" type="submit" disabled={loading}>{loading ? 'Analyzing job...' : 'Analyze job advertisement →'}</button><button className="secondary-button" type="button" onClick={() => { setForm({ ...emptyForm }); setResult(null); setError('') }} disabled={loading}>Clear</button></div>
              <p className="form-hint">At least one field is required. More details may provide a more informative assessment.</p>
            </form>
            <aside className="panel-card result-panel" aria-live="polite"><p className="eyebrow">ANALYSIS REPORT</p><h2>Your results</h2>
              {!result && !loading && <div className="empty-state"><span className="empty-state-icon">⌕</span><h3>Ready when you are</h3><p>Submit a job advertisement to see its model-generated classification and risk score.</p></div>}
              {loading && <div className="loading-state"><div className="spinner" /><h3>Analyzing advertisement</h3><p>Sending job text to the ScamShield model...</p></div>}
              {result && <div className="prediction-result"><span className={`prediction-pill ${result.prediction}`}>{result.prediction === 'fraudulent' ? '⚠ Potential warning signs' : '✓ Classified as legitimate'}</span>
                <div className="score-block"><p>Fraudulent-class model score</p><div className="score-number">{Number(result.risk_score).toFixed(2)}<span>/ 100</span></div><div className="score-track"><div className={`score-fill ${result.risk_level.toLowerCase()}`} style={{ width: `${Math.min(100, Math.max(0, result.risk_score))}%` }} /></div></div>
                <div className="result-level"><span>Display risk level</span><strong className={`level-${result.risk_level.toLowerCase()}`}>{result.risk_level}</strong></div><div className="result-disclaimer"><strong>Important context</strong><p>{result.disclaimer}</p><p>{result.risk_level_note}</p></div>
              </div>}
              <div className="how-it-works"><p className="eyebrow">HOW IT WORKS</p><h3>From job text to a model estimate</h3><ol className="process-list"><li><span>1</span><div><strong>Provide details</strong><p>Paste available job-ad information.</p></div></li><li><span>2</span><div><strong>Analyze text patterns</strong><p>The TF-IDF and logistic regression pipeline scores the text.</p></div></li><li><span>3</span><div><strong>Review the result</strong><p>Use the output as one signal, not as a final verdict.</p></div></li></ol></div>
            </aside>
          </div>
        </section>}

        {page === 'history' && <section className="page-content">
          <div className="page-intro history-intro"><p>Review saved job advertisements and their model-generated assessments.</p><button className="secondary-button" type="button" onClick={() => void fetchHistory()} disabled={historyLoading}>{historyLoading ? 'Refreshing...' : '↻ Refresh history'}</button></div>
          {historyError && <div className="error-message" role="alert">{historyError}</div>}
          <section className="panel-card"><div className="history-filters"><div className="search-field"><span>⌕</span><input aria-label="Search history" value={search} onChange={event => setSearch(event.target.value)} placeholder="Search title, company, or description..." /></div><label className="filter-label">Risk level<select value={riskFilter} onChange={event => setRiskFilter(event.target.value as RiskFilter)}><option>All</option><option>High</option><option>Moderate</option><option>Low</option></select></label></div>
            {historyLoading ? <p className="empty-note">Loading saved analyses...</p> : filteredHistory.length === 0 ? <div className="empty-state"><span className="empty-state-icon">◷</span><h3>{history.length ? 'No matching analyses' : 'No saved analyses yet'}</h3><p>{history.length ? 'Try another search term or risk filter.' : 'Analyze a job advertisement and it will appear here.'}</p><button className="primary-button" type="button" onClick={() => setPage('analyzer')}>Open Job Analyzer</button></div> : <div className="recent-list">{filteredHistory.map(item => <HistoryRow key={item.id} item={item} onLoad={() => loadAdvertisement(item)} />)}</div>}
          </section><p className="dashboard-disclaimer">A low score does not guarantee that a job is genuine, and a high score does not prove fraud.</p>
        </section>}

        <footer className="site-footer"><p>🛡️ ScamShield · Machine-learning job-ad risk estimator</p><p>Historical-data estimates only. Always verify employers and never pay to secure a job.</p></footer>
      </section>
    </main>
  )
}

function StatCard({ label, value, caption, tone = '' }: { label: string; value: number; caption: string; tone?: string }) {
  return <article className={`stat-card ${tone}`}><div className="stat-card-top"><span>{label}</span><span className="stat-icon">{tone === 'high' ? '⚠' : tone === 'moderate' ? '◈' : tone === 'low' ? '✓' : '▤'}</span></div><strong className="stat-value">{value}</strong><span className="stat-caption">{caption}</span></article>
}

function HistoryRow({ item, onLoad }: { item: HistoryItem; onLoad: () => void }) {
  return <article className="recent-item"><div className="recent-item-main"><h3>{item.title || 'Untitled job advertisement'}</h3><p className="history-date">{item.created_at ? new Date(item.created_at).toLocaleString() : 'Date unavailable'}</p><p className="recent-description">{(item.description || 'No description provided.').slice(0, 180)}{(item.description || '').length > 180 ? '…' : ''}</p></div><div className="recent-item-side"><span className={`history-risk history-risk-${item.risk_level.toLowerCase()}`}>{item.risk_level} · {Number(item.risk_score).toFixed(2)}/100</span><span className={`classification-chip ${item.prediction}`}>{item.prediction === 'fraudulent' ? 'Potential warning signs' : 'Classified as legitimate'}</span><button className="secondary-button" type="button" onClick={onLoad}>Load in analyzer</button></div></article>
}

export default App
