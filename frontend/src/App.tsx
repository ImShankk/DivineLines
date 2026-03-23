import { useState } from 'react';
import axios from 'axios';
import { AlertCircle, ChevronDown, ChevronUp, BarChart3 } from 'lucide-react'; 

const TEAM_NAMES: Record<string, string> = {
  "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
  "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
  "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
  "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
  "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
  "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
  "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
  "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
  "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
  "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards"
};

const TEAMS = Object.keys(TEAM_NAMES);

const StatHeader = ({ awayName, homeName }: any) => (
    <div style={{ display: 'flex', justifyContent: 'space-between', paddingBottom: '0.75rem', marginBottom: '0.75rem', borderBottom: '1px solid #404040' }}>
        <div style={{ width: '35%', textAlign: 'center', color: '#E5E7EB', fontWeight: 'bold', fontSize: '0.85rem' }}>{awayName}</div>
        <div style={{ width: '30%' }}></div>
        <div style={{ width: '35%', textAlign: 'center', color: '#E5E7EB', fontWeight: 'bold', fontSize: '0.85rem' }}>{homeName}</div>
    </div>
);

const StatRow = ({ label, awayVal, homeVal, lowerIsBetter = false, isPlusMinus = false }: any) => {
    const awayNum = Number(awayVal);
    const homeNum = Number(homeVal);
    
    // Calculate who wins the stat category
    const awayBetter = lowerIsBetter ? awayNum < homeNum : awayNum > homeNum;
    const homeBetter = lowerIsBetter ? homeNum < awayNum : homeNum > awayNum;

    const formatVal = (val: number) => {
        if (isPlusMinus && val > 0) return `+${val.toFixed(1)}`;
        return val.toFixed(1);
    };

    return (
        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 0', borderBottom: '1px solid #262626' }}>
            <div style={{ width: '35%', textAlign: 'center', color: awayBetter ? '#22C55E' : '#A3A3A3', fontWeight: awayBetter ? 'bold' : 'normal', fontSize: '1.1rem' }}>
                {formatVal(awayNum)}
            </div>
            <div style={{ width: '30%', textAlign: 'center', color: '#737373', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em', alignSelf: 'center' }}>
                {label}
            </div>
            <div style={{ width: '35%', textAlign: 'center', color: homeBetter ? '#22C55E' : '#A3A3A3', fontWeight: homeBetter ? 'bold' : 'normal', fontSize: '1.1rem' }}>
                {formatVal(homeNum)}
            </div>
        </div>
    );
};

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
          <h1 style={{ color: '#ffffff',fontSize: '2.5rem', fontWeight: 'bold', margin: '0 0 0.5rem 0', letterSpacing: '-0.025em' }}>DivineLines V4</h1>
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
                {TEAMS.map(team => <option key={`away-${team}`} value={team}>{TEAM_NAMES[team]}</option>)}
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
                {TEAMS.map(team => <option key={`home-${team}`} value={team}>{TEAM_NAMES[team]}</option>)}
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

            {/* QUANT EDGE DASHBOARD  (i think it shows now*/}
            {prediction.quant_edge ? (
                <div style={{ backgroundColor: '#050505', border: '1px solid #404040', borderRadius: '0.5rem', padding: '1.5rem', marginBottom: '1.5rem', textAlign: 'left' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', paddingBottom: '0.75rem', borderBottom: '1px solid #262626' }}>
                        <h3 style={{ margin: 0, fontSize: '0.9rem', color: '#A3A3A3', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            LIVE VEGAS ODDS ({prediction.quant_edge.bookmaker})
                        </h3>
                    </div>
                    
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem' }}>
                        
                        {/* Away Team Quant Stats */}
                        <div style={{ flex: 1, padding: '1rem', backgroundColor: '#171717', borderRadius: '0.5rem', border: prediction.quant_edge.away_ev > 0 ? '1px solid #22C55E' : '1px solid #262626' }}>
                            <p style={{ margin: '0 0 0.5rem 0', fontWeight: 'bold', color: '#E5E7EB' }}>{TEAM_NAMES[prediction.away_team]}</p>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', color: '#A3A3A3', marginBottom: '0.25rem' }}>
                                <span>Moneyline:</span>
                                <span style={{ color: '#FFFFFF' }}>{prediction.quant_edge.away_odds > 0 ? `+${prediction.quant_edge.away_odds}` : prediction.quant_edge.away_odds}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', color: '#A3A3A3', marginBottom: '0.25rem' }}>
                                <span>Expected Value:</span>
                                <span style={{ color: prediction.quant_edge.away_ev > 0 ? '#22C55E' : '#EF4444', fontWeight: 'bold' }}>
                                    {prediction.quant_edge.away_ev > 0 ? `+$${prediction.quant_edge.away_ev}` : `-$${Math.abs(prediction.quant_edge.away_ev)}`}
                                </span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', color: '#A3A3A3' }}>
                                <span>Rec. Bet (Q-Kelly):</span>
                                <span style={{ color: prediction.quant_edge.away_kelly > 0 ? '#22C55E' : '#737373' }}>
                                    {prediction.quant_edge.away_kelly}% Bankroll
                                </span>
                            </div>
                        </div>

                        {/* Home Team Quant Stats */}
                        <div style={{ flex: 1, padding: '1rem', backgroundColor: '#171717', borderRadius: '0.5rem', border: prediction.quant_edge.home_ev > 0 ? '1px solid #22C55E' : '1px solid #262626' }}>
                            <p style={{ margin: '0 0 0.5rem 0', fontWeight: 'bold', color: '#E5E7EB' }}>{TEAM_NAMES[prediction.home_team]}</p>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', color: '#A3A3A3', marginBottom: '0.25rem' }}>
                                <span>Moneyline:</span>
                                <span style={{ color: '#FFFFFF' }}>{prediction.quant_edge.home_odds > 0 ? `+${prediction.quant_edge.home_odds}` : prediction.quant_edge.home_odds}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', color: '#A3A3A3', marginBottom: '0.25rem' }}>
                                <span>Expected Value:</span>
                                <span style={{ color: prediction.quant_edge.home_ev > 0 ? '#22C55E' : '#EF4444', fontWeight: 'bold' }}>
                                    {prediction.quant_edge.home_ev > 0 ? `+$${prediction.quant_edge.home_ev}` : `-$${Math.abs(prediction.quant_edge.home_ev)}`}
                                </span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.9rem', color: '#A3A3A3' }}>
                                <span>Rec. Bet (Q-Kelly):</span>
                                <span style={{ color: prediction.quant_edge.home_kelly > 0 ? '#22C55E' : '#737373' }}>
                                    {prediction.quant_edge.home_kelly}% Bankroll
                                </span>
                            </div>
                        </div>

                    </div>
                </div>
            ) : (
                <div style={{ backgroundColor: '#0A0A0A', border: '1px solid #262626', borderRadius: '0.5rem', padding: '1rem', marginBottom: '1.5rem', textAlign: 'center', color: '#737373', fontSize: '0.9rem' }}>
                    Live Sportsbook Odds are currently unavailable for this matchup.
                </div>
            )}
            {/*  */}

            <button 
                onClick={() => setShowAdvanced(!showAdvanced)}
                style={{ background: 'none', border: 'none', color: '#9CA3AF', display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', gap: '0.5rem', cursor: 'pointer', padding: '0.5rem', fontWeight: '600' }}
            >
                <BarChart3 size={18} />
                {showAdvanced ? 'Hide Advanced Metrics' : 'View Advanced Metrics'}
                {showAdvanced ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
            </button>

            {/* The Advanced Metrics Dropdown using the new Python payload --- */}
            {showAdvanced && prediction.metrics && prediction.metrics.away_stats && (
                <div style={{ marginTop: '1.5rem', paddingTop: '1.5rem', borderTop: '1px solid #262626', textAlign: 'left' }}>
                    
                    {/* H2H Banner */}
                {prediction.metrics.h2h_stats ? (
                    <div style={{ backgroundColor: '#0A0A0A', padding: '1rem', borderRadius: '0.5rem', border: '1px solid #404040', marginBottom: '1.5rem', textAlign: 'center' }}>
                        <p style={{ color: '#A3A3A3', fontSize: '0.85rem', fontWeight: 'bold', margin: '0 0 0.5rem 0', letterSpacing: '0.05em' }}>LAST HEAD-TO-HEAD</p>
                        <p style={{ color: '#E5E7EB', fontSize: '1.1rem', fontWeight: '600', margin: 0 }}>
                            {prediction.metrics.last_h2h}
                        </p>

                        {/* H2H Stats Grid */}
                        <StatHeader awayName={TEAM_NAMES[prediction.away_team]} homeName={TEAM_NAMES[prediction.home_team]} />
                        
                        <StatRow label="Points" awayVal={prediction.metrics.h2h_stats.away.pts} homeVal={prediction.metrics.h2h_stats.home.pts} />
                        <StatRow label="Points Allowed" awayVal={prediction.metrics.h2h_stats.away.opp_pts} homeVal={prediction.metrics.h2h_stats.home.opp_pts} lowerIsBetter={true} />
                        <StatRow label="3PT Made" awayVal={prediction.metrics.h2h_stats.away.fg3m} homeVal={prediction.metrics.h2h_stats.home.fg3m} />
                        <StatRow label="Rebounds" awayVal={prediction.metrics.h2h_stats.away.reb} homeVal={prediction.metrics.h2h_stats.home.reb} />
                        <StatRow label="Assists" awayVal={prediction.metrics.h2h_stats.away.ast} homeVal={prediction.metrics.h2h_stats.home.ast} />
                        <StatRow label="Turnovers" awayVal={prediction.metrics.h2h_stats.away.tov} homeVal={prediction.metrics.h2h_stats.home.tov} lowerIsBetter={true} />
                        <StatRow label="Fouls" awayVal={prediction.metrics.h2h_stats.away.pf} homeVal={prediction.metrics.h2h_stats.home.pf} lowerIsBetter={true} />
                    
                    </div>
                    ) : (
                    <div style={{ backgroundColor: '#0A0A0A', padding: '1rem', borderRadius: '0.5rem', border: '1px solid #404040', marginBottom: '1.5rem', textAlign: 'center' }}>
                            <p style={{ color: '#A3A3A3', fontSize: '0.85rem', fontWeight: 'bold', margin: '0 0 0.5rem 0', letterSpacing: '0.05em' }}>LAST HEAD-TO-HEAD</p>
                            <p style={{ color: '#E5E7EB', fontSize: '1.1rem', fontWeight: '600', margin: 0 }}>No matchups yet this season.</p>
                        </div>
                    )}


                    {/* Tale of the Tape Grid */}
                    <div style={{ backgroundColor: '#0A0A0A', padding: '1.5rem', borderRadius: '0.5rem', border: '1px solid #404040' }}>
                        <p style={{ color: '#A3A3A3', fontSize: '0.85rem', fontWeight: 'bold', margin: '0 0 1.5rem 0', letterSpacing: '0.05em', textAlign: 'center' }}>ROLLING 10-GAME AVERAGES</p>
                        
                        {/* We use the custom StatRow component to do the heavy lifting here */}
                        <StatRow label="Points" awayVal={prediction.metrics.away_stats.pts} homeVal={prediction.metrics.home_stats.pts} />
                        <StatRow label="Points Allowed" awayVal={prediction.metrics.away_stats.opp_pts} homeVal={prediction.metrics.home_stats.opp_pts} lowerIsBetter={true} />
                        <StatRow label="3PT Made" awayVal={prediction.metrics.away_stats.fg3m} homeVal={prediction.metrics.home_stats.fg3m} />
                        <StatRow label="Rebounds" awayVal={prediction.metrics.away_stats.reb} homeVal={prediction.metrics.home_stats.reb} />
                        <StatRow label="Assists" awayVal={prediction.metrics.away_stats.ast} homeVal={prediction.metrics.home_stats.ast} />
                        <StatRow label="Turnovers" awayVal={prediction.metrics.away_stats.tov} homeVal={prediction.metrics.home_stats.tov} lowerIsBetter={true} />
                        <StatRow label="Offensive Rtg" awayVal={prediction.metrics.away_stats.ortg} homeVal={prediction.metrics.home_stats.ortg} />
                        <StatRow label="Defensive Rtg" awayVal={prediction.metrics.away_stats.drtg} homeVal={prediction.metrics.home_stats.drtg} lowerIsBetter={true} />
                        <StatRow label="Net Rating" awayVal={prediction.metrics.away_stats.net_rating} homeVal={prediction.metrics.home_stats.net_rating} isPlusMinus={true} />
                        <StatRow label="Pace (Poss)" awayVal={prediction.metrics.away_stats.pace} homeVal={prediction.metrics.home_stats.pace} />
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