import { useState, useEffect } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

function DrillPage() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()

  const motion = searchParams.get('motion') || 'Social media does more harm than good'
  const position = searchParams.get('position') || 'for'

  const [currentClaim, setCurrentClaim] = useState(null)
  const [claimPosition, setClaimPosition] = useState(null)
  const [rebuttal, setRebuttal] = useState('')
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [lastScore, setLastScore] = useState(null)
  const [attemptCount, setAttemptCount] = useState(0)

  // Start drill - get first claim
  useEffect(() => {
    startDrill()
  }, [])

  const startDrill = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE}/v1/drills/rebuttal/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          motion: motion,
          user_position: position,
        }),
      })

      if (!response.ok) throw new Error('Failed to start drill')

      const data = await response.json()
      setCurrentClaim(data.claim)
      setClaimPosition(data.claim_position)
    } catch (error) {
      console.error('Error starting drill:', error)
      alert('Failed to start drill. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const submitRebuttal = async () => {
    if (!rebuttal.trim()) {
      alert('Please enter your rebuttal')
      return
    }

    setSubmitting(true)
    try {
      const response = await fetch(`${API_BASE}/v1/drills/rebuttal/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          motion: motion,
          claim: currentClaim,
          claim_position: claimPosition,
          rebuttal: rebuttal.trim(),
        }),
      })

      if (!response.ok) throw new Error('Failed to submit rebuttal')

      const data = await response.json()
      setLastScore(data)
      setCurrentClaim(data.next_claim)
      setClaimPosition(data.next_claim_position)
      setRebuttal('')
      setAttemptCount(prev => prev + 1)

      // Scroll to score feedback
      setTimeout(() => {
        document.querySelector('.drill-score')?.scrollIntoView({ behavior: 'smooth' })
      }, 100)
    } catch (error) {
      console.error('Error submitting rebuttal:', error)
      // NEVER expose error.message - could contain prompts during network failures
      alert('Failed to submit rebuttal. Please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  const getScoreColor = (score) => {
    if (score >= 8) return '#22c55e'
    if (score >= 6) return '#eab308'
    if (score >= 4) return '#f97316'
    return '#ef4444'
  }

  if (loading && !currentClaim) {
    return (
      <div className="app drill-mode">
        <div className="drill-container">
          <p>Loading drill...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="app drill-mode">
      <button
        className="return-to-landing"
        onClick={() => navigate('/')}
        title="Return to home"
      >
        ← Home
      </button>

      <div className="drill-container">
        <div className="drill-header">
          <h1>Rebuttal Drill</h1>
          <p className="drill-subtitle">Practice refuting claims on: <strong>{motion}</strong></p>
          <p className="drill-info">
            You argued <strong>{position.toUpperCase()}</strong> •
            Rebut claims from the <strong>{claimPosition?.toUpperCase()}</strong> side •
            Attempts: {attemptCount}
          </p>
        </div>

        {lastScore && (
          <div className="drill-score">
            <h3>Last Attempt Score</h3>
            <div className="score-display-mini">
              <div
                className="score-number-mini"
                style={{ color: getScoreColor(lastScore.overall_score) }}
              >
                {lastScore.overall_score.toFixed(1)}/10
              </div>
              <div className="score-metrics-mini">
                <div>
                  <span>Refutation</span>
                  <strong>{lastScore.metrics.refutation_quality.toFixed(1)}</strong>
                </div>
                <div>
                  <span>Evidence</span>
                  <strong>{lastScore.metrics.evidence_examples.toFixed(1)}</strong>
                </div>
                <div>
                  <span>Impact</span>
                  <strong>{lastScore.metrics.impact_comparison.toFixed(1)}</strong>
                </div>
              </div>
            </div>
            <p className="drill-feedback">{lastScore.feedback}</p>
          </div>
        )}

        <div className="drill-claim-box">
          <h3>Claim to Rebut</h3>
          <div className="claim-content">
            <span className="claim-position-tag">{claimPosition?.toUpperCase()}</span>
            <p>{currentClaim}</p>
          </div>
        </div>

        <div className="drill-input-section">
          <h3>Your Rebuttal</h3>
          <textarea
            className="drill-textarea"
            value={rebuttal}
            onChange={(e) => setRebuttal(e.target.value)}
            placeholder="Write your rebuttal here... Focus on: (1) Negating/mitigating the claim, (2) Using evidence/examples, (3) Comparing impacts"
            rows={8}
            disabled={submitting}
          />
          <button
            className="btn-primary btn-large"
            onClick={submitRebuttal}
            disabled={submitting || !rebuttal.trim()}
          >
            {submitting ? 'Scoring...' : 'Submit Rebuttal'}
          </button>
        </div>

        <div className="drill-tips">
          <h4>Tips for Strong Rebuttals</h4>
          <ul>
            <li><strong>Negate first:</strong> Show why the claim isn't true or doesn't work</li>
            <li><strong>Use examples:</strong> Counter with specific real-world evidence</li>
            <li><strong>Weigh impacts:</strong> Explain why your refutation matters more</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

export default DrillPage
