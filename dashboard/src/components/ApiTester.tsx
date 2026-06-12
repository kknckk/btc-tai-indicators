"use client";
import React, { useState } from 'react';
import { Play, Code } from 'lucide-react';

interface ApiTesterProps {
  initialMetric?: string;
}

export default function ApiTester({ initialMetric = 'MVRV_Z' }: ApiTesterProps) {
  const [metric, setMetric] = useState(initialMetric);
  const [startDate, setStartDate] = useState('2024-01-01');
  const [endDate, setEndDate] = useState('2024-05-01');
  const [response, setResponse] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const testApi = async () => {
    setLoading(true);
    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "https://btc-api-203925818774.europe-central2.run.app";
      const res = await fetch(`${baseUrl}/api/v1/metrics/values?metrics=${metric}&start_date=${startDate}&end_date=${endDate}`);
      const data = await res.json();
      setResponse(data);
    } catch (err: any) {
      setResponse({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full bg-slate-900 rounded-xl p-6 border border-slate-800 shadow-xl">
      <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
        <Code className="text-blue-500" />
        API Tester
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-slate-400 mb-1">Metric</label>
          <input 
            type="text" 
            value={metric}
            onChange={e => setMetric(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-400 mb-1">Start Date</label>
          <input 
            type="date" 
            value={startDate}
            onChange={e => setStartDate(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-400 mb-1">End Date</label>
          <input 
            type="date" 
            value={endDate}
            onChange={e => setEndDate(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
          />
        </div>
        <div className="flex items-end">
          <button 
            onClick={testApi}
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
          >
            <Play size={18} />
            {loading ? 'Fetching...' : 'Send Request'}
          </button>
        </div>
      </div>

      <div className="bg-black/50 rounded-lg p-4 font-mono text-sm overflow-x-auto border border-slate-800 min-h-[150px] max-h-[300px] overflow-y-auto">
        {response ? (
          <pre className="text-emerald-400">
            {JSON.stringify(response, null, 2)}
          </pre>
        ) : (
          <span className="text-slate-500">// Response will appear here</span>
        )}
      </div>
    </div>
  );
}
