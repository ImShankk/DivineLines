import { useState } from 'react';
import axios from 'axios';
import { Activity, Trophy, AlertCircle } from 'lucide-react';

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

  const handlePredict = async () => {
    if (homeTeam === awayTeam) {
      setError("A team cannot play itself.");
      return;
    }
    
    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      // This reaches out to your Python FastAPI server
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
    <div style={{ minHeight: '100vh', backgroundColor: '#0f172a', color: '#f8fafc', fontFamily: 'system-ui, sans-serif', padding: '3rem 1rem' }}>
      <div style={{ maxWidth: '600px', margin: '0 auto' }}>
        
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <Activity size={48} color="#38bdf8" style={{ margin: '0 auto', marginBottom: '1rem' }} />
          <h1 style={{ fontSize: '2.5rem', fontWeight: 'bold', margin: '0 0 0.5rem 0' }}>DivineLines V4</h1>
          <p style={{ color: '#94a3b8', fontSize: '1.1rem' }}>Predictive Math & Matchup Engine</p>
        </div>

        {/* Input Panel */}
        <div style={{ backgroundColor: '#1e293b', padding: '2rem', borderRadius: '1rem', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', marginBottom: '2rem' }}>
            
            <div style={{ flex: 1 }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', color: '#94a3b8', fontWeight: 'bold', fontSize: '0.9rem' }}>AWAY TEAM</label>
              <select 
                value={awayTeam} 
                onChange={(e) => setAwayTeam(e.target.value)}
                style={{ width: '100%', padding: '0.75rem', backgroundColor: '#0f172a', color: 'white', border: '1px solid #334155', borderRadius: '0.5rem', fontSize: '1rem' }}
              >
                {TEAMS.map(team => <option key={`away-${team}`} value={team}>{team}</option>)}
              </select>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', color: '#64748b', fontWeight: 'bold', paddingTop: '1.5rem' }}>@</div>

            <div style={{ flex: 1 }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', color: '#94a3b8', fontWeight: 'bold', fontSize: '0.9rem' }}>HOME TEAM</label>
              <select 
                value={homeTeam} 
                onChange={(e) => setHomeTeam(e.target.value)}
                style={{ width: '100%', padding: '0.75rem', backgroundColor: '#0f172a', color: 'white', border: '1px solid #334155', borderRadius: '0.5rem', fontSize: '1rem' }}
              >
                {TEAMS.map(team => <option key={`home-${team}`} value={team}>{team}</option>)}
              </select>
            </div>
            
          </div>

          <button 
            onClick={handlePredict}
            disabled={loading}
            style={{ width: '100%', padding: '1rem', backgroundColor: '#0284c7', color: 'white', border: 'none', borderRadius: '0.5rem', fontSize: '1.1rem', fontWeight: 'bold', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1 }}
          >
            {loading ? 'Consulting the Engine...' : 'Run Prediction'}
          </button>
        </div>

        {/* Error State */}
        {error && (
          <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#7f1d1d', color: '#fca5a5', borderRadius: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertCircle size={20} />
            {error}
          </div>
        )}

        {/* Results Panel */}
        {prediction && !prediction.message.includes("Engine Error") && (
          <div style={{ marginTop: '2rem', padding: '2rem', backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '1rem', textAlign: 'center', boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.5)' }}>
            <h2 style={{ fontSize: '1.2rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '1.5rem' }}>Model Projection</h2>
            
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '3rem', marginBottom: '2rem' }}>
                {/* Away Team */}
                <div style={{ textAlign: 'center' }}>
                    <p style={{ fontSize: '2.5rem', fontWeight: 'bold', color: prediction.home_win_probability < 50 ? '#10b981' : '#475569', transition: 'color 0.3s' }}>
                      {prediction.away_team}
                    </p>
                    <p style={{ color: prediction.home_win_probability < 50 ? '#34d399' : '#64748b', fontSize: '1.2rem', fontWeight: 'bold' }}>
                      {(100 - prediction.home_win_probability).toFixed(1)}%
                    </p>
                </div>

                <div style={{ color: '#334155', fontSize: '1.5rem', fontWeight: 'bold' }}>VS</div>

                {/* Home Team */}
                <div style={{ textAlign: 'center' }}>
                    <p style={{ fontSize: '2.5rem', fontWeight: 'bold', color: prediction.home_win_probability >= 50 ? '#10b981' : '#475569', transition: 'color 0.3s' }}>
                      {prediction.home_team}
                    </p>
                    <p style={{ color: prediction.home_win_probability >= 50 ? '#34d399' : '#64748b', fontSize: '1.2rem', fontWeight: 'bold' }}>
                      {prediction.home_win_probability.toFixed(1)}%
                    </p>
                </div>
            </div>

            <div style={{ backgroundColor: '#0f172a', padding: '1rem', borderRadius: '0.5rem', color: '#cbd5e1', borderLeft: '4px solid #38bdf8', textAlign: 'left' }}>
                <strong>Analysis: </strong> {prediction.message}
            </div>
          </div>
        )}

        {/* Error Handling for Engine Fails */}
        {prediction && prediction.message.includes("Engine Error") && (
          <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#7f1d1d', color: '#fca5a5', borderRadius: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertCircle size={20} />
            {prediction.message}
          </div>
        )}

      </div>
    </div>
  );
}

export default App;