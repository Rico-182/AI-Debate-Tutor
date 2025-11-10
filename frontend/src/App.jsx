import { useState, useEffect, useRef } from 'react'

const API_BASE = 'http://localhost:8000'

function App() {
  const [debateId, setDebateId] = useState(null)
  const [debate, setDebate] = useState(null)
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [score, setScore] = useState(null)
  const [scoring, setScoring] = useState(false)
  const [scoreError, setScoreError] = useState(null)
  
  // Setup state
  const [topic, setTopic] = useState('')
  const [position, setPosition] = useState('for') // 'for' or 'against'
  const [numRounds, setNumRounds] = useState(3)
  const [setupComplete, setSetupComplete] = useState(false)
  
  // Input state
  const [argument, setArgument] = useState('')
  const [audioFile, setAudioFile] = useState(null)
  const inputRef = useRef(null)

  const fetchScore = async (targetId = debateId, compute = false) => {
    if (!targetId) return null
    setScoring(true)
    try {
      const method = compute ? 'POST' : 'GET'
      const response = await fetch(`${API_BASE}/v1/debates/${targetId}/score`, {
        method,
        headers: { 'Content-Type': 'application/json' },
      })

      if (response.ok) {
        const data = await response.json()
        setScore(data)
        setScoreError(null)
        return data
      }

      if (response.status === 404 && !compute) {
        // Score not yet generated; compute it
        return await fetchScore(targetId, true)
      }

      const errorText = await response.text()
      throw new Error(errorText || 'Failed to fetch score')
    } catch (error) {
      console.error('Error fetching score:', error)
      setScoreError(error.message)
      return null
    } finally {
      setScoring(false)
    }
  }
  useEffect(() => {
    if (debateId) {
      fetchDebate()
      const interval = setInterval(fetchDebate, 2000) // Poll every 2 seconds
      return () => clearInterval(interval)
    }
  }, [debateId])

  const fetchDebate = async (targetId = debateId) => {
    if (!targetId) return null
    try {
      const response = await fetch(`${API_BASE}/v1/debates/${targetId}`)
      if (!response.ok) throw new Error('Failed to fetch debate')
      const data = await response.json()
      setDebate(data)
      setMessages(data.messages || [])

      if (
        data.status === 'completed' &&
        targetId === debateId &&
        (!score || score.debate_id !== targetId) &&
        !scoring
      ) {
        fetchScore(targetId, false)
      }
      return data
    } catch (error) {
      console.error('Error fetching debate:', error)
      return null
    }
  }

  const startDebate = async () => {
    if (!topic.trim()) {
      alert('Please enter a debate topic')
      return
    }

    setLoading(true)
    try {
      // Determine starter based on position
      // If user is "for", user starts; if "against", assistant starts (arguing for)
      const starter = position === 'for' ? 'user' : 'assistant'
      
      const response = await fetch(`${API_BASE}/v1/debates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_rounds: numRounds,
          starter,
          title: `${topic} (User: ${position}, You take the opposite position)`,
        }),
      })

      if (!response.ok) throw new Error('Failed to create debate')
      const data = await response.json()
      setDebateId(data.id)
      setDebate(data)
      setSetupComplete(true)
      setScore(null)
      setScoreError(null)

      // If assistant starts, generate their first turn
      if (starter === 'assistant') {
        setTimeout(() => generateAITurn(data.id), 500)
      }
    } catch (error) {
      alert('Failed to start debate: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const transcribeAudio = async (file) => {
    const formData = new FormData()
    formData.append('file', file)
    try {
      const response = await fetch(`${API_BASE}/v1/transcribe`, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) throw new Error('Transcription failed')
      const data = await response.json()
      return data.text
    } catch (error) {
      alert('Transcription failed: ' + error.message)
      return null
    }
  }

  const submitArgument = async () => {
    if (!debateId || !argument.trim()) {
      alert('Please enter your argument')
      return
    }

    setSubmitting(true)
    try {
      let finalContent = argument.trim()

      // If audio file exists, transcribe it
      if (audioFile && !finalContent) {
        const transcribed = await transcribeAudio(audioFile)
        if (!transcribed) {
          setSubmitting(false)
          return
        }
        finalContent = transcribed
        setArgument(transcribed)
      }

      if (!finalContent) {
        alert('Please provide text or an audio file')
        setSubmitting(false)
        return
      }

      const response = await fetch(`${API_BASE}/v1/debates/${debateId}/turns`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          speaker: 'user',
          content: finalContent,
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText)
      }

      const turnData = await response.json()
      console.log('Turn submitted, response:', turnData)
      
      setArgument('')
      setAudioFile(null)
      
      // Refresh debate state to get latest messages
      const updatedDebate = await fetchDebate()
      console.log('Updated debate after turn:', updatedDebate)
      
      // Auto-generate AI response if it's the assistant's turn
      if (turnData.status === 'active' && turnData.next_speaker === 'assistant') {
        console.log('Triggering AI turn generation...')
        // Small delay to ensure UI updates
        setTimeout(() => {
          generateAITurn(debateId)
        }, 800)
      } else if (turnData.status === 'completed') {
        await fetchScore(debateId, false)
      }
    } catch (error) {
      alert('Failed to submit argument: ' + error.message)
    } finally {
      setSubmitting(false)
    }
  }

  const generateAITurn = async (targetId = debateId) => {
    if (!targetId) return

    setLoading(true)
    try {
      const response = await fetch(`${API_BASE}/v1/debates/${targetId}/auto-turn`, {
        method: 'POST',
      })
      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText)
      }
      
      // Get the AI response data - this contains the message content
      const aiTurnData = await response.json()
      console.log('AI turn generated:', aiTurnData)
      
      // Immediately refresh to show the new message
      const updated = await fetchDebate(targetId)
      if (updated?.status === 'completed') {
        await fetchScore(targetId, false)
      }
    } catch (error) {
      console.error('Error generating AI turn:', error)
      alert('Failed to generate AI response: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const finishDebate = async () => {
    if (!debateId) return
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE}/v1/debates/${debateId}/finish`, {
        method: 'POST',
      })
      if (!response.ok) throw new Error('Failed to finish debate')
      await fetchDebate()
      await fetchScore(debateId, true)
    } catch (error) {
      alert('Failed to finish debate: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const resetDebate = () => {
    setDebateId(null)
    setDebate(null)
    setMessages([])
    setSetupComplete(false)
    setArgument('')
    setAudioFile(null)
    setScore(null)
    setScoreError(null)
  }

  // Setup screen
  if (!setupComplete) {
    return (
      <div className="app setup-mode">
        <div className="setup-container">
          <div className="setup-card">
            <h1>DebateLab</h1>
            <p className="subtitle">Practice your debating skills with an AI opponent</p>
            
            <div className="form-group">
              <label>Debate Topic</label>
              <input
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="e.g., Social media does more harm than good"
                className="input-large"
                onKeyPress={(e) => e.key === 'Enter' && startDebate()}
              />
            </div>

            <div className="form-group">
              <label>Your Position</label>
              <div className="position-buttons">
                <button
                  className={position === 'for' ? 'position-btn active' : 'position-btn'}
                  onClick={() => setPosition('for')}
                >
                  For
                </button>
                <button
                  className={position === 'against' ? 'position-btn active' : 'position-btn'}
                  onClick={() => setPosition('against')}
                >
                  Against
                </button>
              </div>
            </div>

            <div className="form-group">
              <label>Number of Rounds</label>
              <select
                value={numRounds}
                onChange={(e) => setNumRounds(parseInt(e.target.value))}
                className="input-large"
              >
                <option value={1}>1 Round</option>
                <option value={2}>2 Rounds</option>
                <option value={3}>3 Rounds</option>
              </select>
            </div>

            <button
              className="btn-primary btn-large"
              onClick={startDebate}
              disabled={loading || !topic.trim()}
            >
              {loading ? 'Starting...' : 'Start Debate'}
            </button>
          </div>
        </div>
      </div>
    )
  }

  // Debate screen
  return (
    <div className="app debate-mode">
      <header className="debate-header">
        <div className="header-content">
          <div>
            <h2>{topic}</h2>
            <p className="header-subtitle">
              Your position: <strong>{position === 'for' ? 'FOR' : 'AGAINST'}</strong>
              {' • '}
              Round {debate?.current_round || 1} of {debate?.num_rounds || 3}
              {' • '}
              <span className={`status ${debate?.status || 'active'}`}>
                {debate?.status === 'active' ? 'Active' : 'Completed'}
              </span>
            </p>
          </div>
          <button className="btn-secondary" onClick={resetDebate}>
            New Debate
          </button>
        </div>
      </header>

      <main className="debate-main">
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="empty-state">
              <p>The debate is starting. {debate?.next_speaker === 'assistant' ? 'Waiting for AI...' : 'Make your first argument!'}</p>
            </div>
          )}
          
          {messages.map((message, index) => (
            <div
              key={message.id}
              className={`message ${message.speaker === 'user' ? 'message-user' : 'message-ai'}`}
            >
              <div className="message-header">
                <span className="message-speaker">
                  {message.speaker === 'user' 
                    ? `You (${position === 'for' ? 'FOR' : 'AGAINST'})`
                    : `AI (${position === 'for' ? 'AGAINST' : 'FOR'})`
                  }
                </span>
                <span className="message-round">Round {message.round_no}</span>
              </div>
              <div className="message-content">{message.content}</div>
            </div>
          ))}
          
          {loading && debate?.status === 'active' && (
            <div className="message message-ai">
              <div className="message-header">
                <span className="message-speaker">
                  AI ({position === 'for' ? 'AGAINST' : 'FOR'})
                </span>
              </div>
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
        </div>

        {debate?.status === 'active' && debate?.next_speaker === 'user' && (
          <div className="input-container">
            <div className="input-wrapper">
              <textarea
                ref={inputRef}
                value={argument}
                onChange={(e) => setArgument(e.target.value)}
                placeholder="Type your argument here..."
                className="argument-input"
                rows={4}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                    submitArgument()
                  }
                }}
                disabled={submitting}
              />
              <div className="input-actions">
                <label className="file-upload-btn">
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={(e) => setAudioFile(e.target.files[0] || null)}
                    style={{ display: 'none' }}
                  />
                  🎤 Audio
                </label>
                {audioFile && (
                  <span className="file-name">{audioFile.name}</span>
                )}
                <button
                  className="btn-primary"
                  onClick={submitArgument}
                  disabled={submitting || !argument.trim()}
                >
                  {submitting ? 'Submitting...' : 'Send'}
                </button>
              </div>
              <p className="input-hint">Press Cmd/Ctrl + Enter to send</p>
            </div>
          </div>
        )}

        {debate?.status === 'completed' && (
          <div className="debate-ended">
            <p>Debate completed! Review your score or start a new debate to continue practicing.</p>
            {score ? (
              <div className="score-card">
                <h3>Debate Score</h3>
                <p className="overall-score">{score.overall.toFixed(1)} / 100</p>
                <div className="score-metrics">
                  <div>
                    <span>Clarity & Delivery</span>
                    <strong>{score.metrics?.clarity?.toFixed(1) ?? '–'}</strong>
                    <small>{score.clarity_feedback}</small>
                  </div>
                  <div>
                    <span>Structure & Organization</span>
                    <strong>{score.metrics?.structure?.toFixed(1) ?? '–'}</strong>
                    <small>{score.structure_feedback}</small>
                  </div>
                  <div>
                    <span>Engagement & Clash</span>
                    <strong>{score.metrics?.engagement?.toFixed(1) ?? '–'}</strong>
                    <small>{score.engagement_feedback}</small>
                  </div>
                  <div>
                    <span>Balance & Completion</span>
                    <strong>{score.metrics?.balance?.toFixed(1) ?? '–'}</strong>
                    <small>{score.balance_feedback}</small>
                  </div>
                </div>
                <div className="score-feedback">
                  <h4>Judge Feedback</h4>
                  <p>{score.feedback}</p>
                </div>
              </div>
            ) : (
              <div className="score-card">
                <p className="muted">
                  {scoring
                    ? 'Calculating score...'
                    : scoreError
                      ? `Unable to load score: ${scoreError}`
                      : 'Score not available yet.'}
                </p>
                <button
                  className="btn-secondary"
                  onClick={() => fetchScore(debateId, true)}
                  disabled={scoring}
                >
                  {scoring ? 'Scoring...' : 'Calculate Score'}
                </button>
              </div>
            )}
            <button className="btn-primary" onClick={resetDebate}>
              New Debate
            </button>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
