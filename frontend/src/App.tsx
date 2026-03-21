import { useState } from 'react';
import axios from 'axios';
import { AlertCircle, ChevronDown, ChevronUp, BarChart3 } from 'lucide-react'; 

const TEAMS = [
  "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
  "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
  "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
];

function App() {
  const [homeTeam, setHomeTeam] = useState('BOS');
  const [awayTeam, setAwayTeam] = useState('DAL');
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false); // THE NEW TOGGLE STATE

  const handlePredict = async () => {
    if (homeTeam === awayTeam) {
      setError("A team cannot play itself.");
      return;
    }
    
    setLoading(true);
    setError('');
    setPrediction(null);
    setShowAdvanced(false);

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/predict', {
        home: homeTeam,
        away: awayTeam
      });
      setPrediction(response.data);
    } catch (err) {
      setError("Failed to connect to the DivineLines Engine. Is Python running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#0A0A0A', color: '#FFFFFF', fontFamily: 'system-ui, sans-serif', padding: '3rem 1rem' }}>
      <div style={{ maxWidth: '600px', margin: '0 auto' }}>
        
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <h1 style={{ fontSize: '2.5rem', fontWeight: 'bold', margin: '0 0 0.5rem 0', letterSpacing: '-0.025em' }}>DivineLines V4</h1>
          <p style={{ color: '#A3A3A3', fontSize: '1.1rem' }}>Predictive Math & Matchup Engine</p>
        </div>

        {/* Input Panel */}
        <div style={{ backgroundColor: '#171717', padding: '2rem', borderRadius: '0.75rem', border: '1px solid #262626', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.5)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', marginBottom: '2rem' }}>
            
            <div style={{ flex: 1 }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', color: '#A3A3A3', fontWeight: '600', fontSize: '0.85rem', letterSpacing: '0.05em' }}>AWAY TEAM</label>
              <select 
                value={awayTeam} 
                onChange={(e) => setAwayTeam(e.target.value)}
                style={{ width: '100%', padding: '0.75rem', backgroundColor: '#262626', color: '#FFFFFF', border: '1px solid #404040', borderRadius: '0.5rem', fontSize: '1rem', outline: 'none' }}
              >
                {TEAMS.map(team => <option key={`away-${team}`} value={team}>{team}</option>)}
              </select>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', color: '#525252', fontWeight: 'bold', paddingTop: '1.5rem' }}>@</div>

            <div style={{ flex: 1 }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', color: '#A3A3A3', fontWeight: '600', fontSize: '0.85rem', letterSpacing: '0.05em' }}>HOME TEAM</label>
              <select 
                value={homeTeam} 
                onChange={(e) => setHomeTeam(e.target.value)}
                style={{ width: '100%', padding: '0.75rem', backgroundColor: '#262626', color: '#FFFFFF', border: '1px solid #404040', borderRadius: '0.5rem', fontSize: '1rem', outline: 'none' }}
              >
                {TEAMS.map(team => <option key={`home-${team}`} value={team}>{team}</option>)}
              </select>
            </div>
            
          </div>

          <button 
            onClick={handlePredict}
            disabled={loading}
            style={{ width: '100%', padding: '1rem', backgroundColor: '#2563EB', color: '#FFFFFF', border: 'none', borderRadius: '0.5rem', fontSize: '1.1rem', fontWeight: 'bold', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1, transition: 'opacity 0.2s' }}
          >
            {loading ? 'Consulting the Engine...' : 'Run Prediction'}
          </button>
        </div>

        {/* Error State */}
        {error && (
          <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#450a0a', border: '1px solid #991b1b', color: '#fca5a5', borderRadius: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <AlertCircle size={20} />
            <span style={{ fontWeight: '500' }}>{error}</span>
          </div>
        )}

        {/* Results Panel */}
        {prediction && !prediction.message.includes("Engine Error") && (
          <div style={{ marginTop: '2rem', padding: '2rem', backgroundColor: '#171717', border: '1px solid #2563EB', borderRadius: '0.75rem', textAlign: 'center', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)' }}>
            <h2 style={{ fontSize: '1rem', color: '#A3A3A3', textTransform: 'uppercase', letterSpacing: '0.15em', marginBottom: '2rem' }}>Model Projection</h2>
            
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '2.5rem', marginBottom: '2.5rem' }}>
                <div style={{ textAlign: 'center', flex: 1 }}>
                    <p style={{ fontSize: '3rem', fontWeight: '800', margin: '0 0 0.5rem 0', color: prediction.home_win_probability < 50 ? '#22C55E' : '#FFFFFF' }}>
                      {prediction.away_team}
                    </p>
                    <p style={{ margin: 0, color: prediction.home_win_probability < 50 ? '#22C55E' : '#A3A3A3', fontSize: '1.25rem', fontWeight: 'bold' }}>
                      {(100 - prediction.home_win_probability).toFixed(1)}%
                    </p>
                </div>

                <div style={{ color: '#525252', fontSize: '1.2rem', fontWeight: 'bold' }}>VS</div>

                <div style={{ textAlign: 'center', flex: 1 }}>
                    <p style={{ fontSize: '3rem', fontWeight: '800', margin: '0 0 0.5rem 0', color: prediction.home_win_probability >= 50 ? '#22C55E' : '#FFFFFF' }}>
                      {prediction.home_team}
                    </p>
                    <p style={{ margin: 0, color: prediction.home_win_probability >= 50 ? '#22C55E' : '#A3A3A3', fontSize: '1.25rem', fontWeight: 'bold' }}>
                      {prediction.home_win_probability.toFixed(1)}%
                    </p>
                </div>
            </div>

            <div style={{ backgroundColor: '#262626', padding: '1.25rem', borderRadius: '0.5rem', color: '#E5E7EB', borderLeft: '4px solid #2563EB', textAlign: 'left', fontSize: '1.05rem', lineHeight: '1.5', marginBottom: '1.5rem' }}>
                <strong style={{ color: '#FFFFFF' }}>Analysis: </strong> {prediction.message}
            </div>

            {/* THE NEW ADVANCED TOGGLE BUTTON */}
            <button 
                onClick={() => setShowAdvanced(!showAdvanced)}
                style={{ background: 'none', border: 'none', color: '#9CA3AF', display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', gap: '0.5rem', cursor: 'pointer', padding: '0.5rem', fontWeight: '600' }}
            >
                <BarChart3 size={18} />
                {showAdvanced ? 'Hide Advanced Metrics' : 'View Advanced Metrics'}
                {showAdvanced ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
            </button>

            {/* ADVANCED METRICS DROPDOWN */}
            {showAdvanced && prediction.metrics && (
                <div style={{ marginTop: '1.5rem', paddingTop: '1.5rem', borderTop: '1px solid #262626', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', textAlign: 'left' }}>
                    <div style={{ backgroundColor: '#0A0A0A', padding: '1rem', borderRadius: '0.5rem', border: '1px solid #404040' }}>
                        <p style={{ color: '#A3A3A3', fontSize: '0.85rem', fontWeight: 'bold', margin: '0 0 0.5rem 0' }}>NET RATING EDGE</p>
                        <p style={{ color: prediction.metrics.net_rating_diff > 0 ? '#22C55E' : '#EF4444', fontSize: '1.5rem', fontWeight: 'bold', margin: 0 }}>
                            {prediction.metrics.net_rating_diff > 0 ? '+' : ''}{prediction.metrics.net_rating_diff}
                        </p>
                        <p style={{ color: '#525252', fontSize: '0.8rem', marginTop: '0.25rem' }}>Home vs Away</p>
                    </div>

                    <div style={{ backgroundColor: '#0A0A0A', padding: '1rem', borderRadius: '0.5rem', border: '1px solid #404040' }}>
                        <p style={{ color: '#A3A3A3', fontSize: '0.85rem', fontWeight: 'bold', margin: '0 0 0.5rem 0' }}>PACE DIFFERENTIAL</p>
                        <p style={{ color: '#38BDF8', fontSize: '1.5rem', fontWeight: 'bold', margin: 0 }}>
                            {prediction.metrics.pace_diff > 0 ? '+' : ''}{prediction.metrics.pace_diff}
                        </p>
                        <p style={{ color: '#525252', fontSize: '0.8rem', marginTop: '0.25rem' }}>Possessions per 48m</p>
                    </div>
                </div>
            )}

          </div>
        )}

      </div>
    </div>
  );
}

export default App;