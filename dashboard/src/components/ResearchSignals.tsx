"use client";

import React, { useEffect, useState } from "react";
import { AlertTriangle, TrendingUp, CheckCircle, Clock } from "lucide-react";

export function ResearchSignals() {
  const [strategy, setStrategy] = useState<any>(null);
  const [ewi, setEwi] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSignals = async () => {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      try {
        const [stratRes, ewiRes] = await Promise.all([
          fetch(`${baseUrl}/api/v1/research/strategy/status`).catch(() => null),
          fetch(`${baseUrl}/api/v1/research/ewi/current`).catch(() => null)
        ]);

        if (stratRes && stratRes.ok) {
          const stratData = await stratRes.json();
          setStrategy(stratData);
        }

        if (ewiRes && ewiRes.ok) {
          const ewiData = await ewiRes.json();
          setEwi(ewiData);
        }
      } catch (e) {
        console.error("Failed to fetch research signals", e);
      } finally {
        setLoading(false);
      }
    };
    fetchSignals();
  }, []);

  if (loading) {
    return (
      <div className="w-full h-24 bg-slate-900 rounded-xl flex items-center justify-center border border-slate-800 shadow-xl">
        <div className="text-slate-400 text-sm">Ładowanie sygnałów badawczych...</div>
      </div>
    );
  }

  return (
    <section className="space-y-6">
      <h2 className="text-2xl font-bold text-white border-b border-slate-800 pb-2 flex items-center gap-2">
        <span className="text-blue-500">⚛</span> Research Signals (v2)
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Strategy Signal */}
        <div className="bg-slate-900 p-6 rounded-xl border border-slate-800 shadow-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-semibold text-slate-200">On-Chain Strategy (Paper 4)</h3>
              <TrendingUp className="text-blue-400 w-5 h-5" />
            </div>
            <p className="text-sm text-slate-400 mb-4">Fixed Literature threshold variant</p>
            
            {strategy?.status === 'ok' ? (
              <div className="space-y-2">
                <div className="flex justify-between items-center bg-slate-950 p-3 rounded-lg">
                  <span className="text-slate-400">Position</span>
                  <span className={`font-bold ${strategy.position === 'Long' ? 'text-emerald-400' : 'text-slate-300'}`}>
                    {strategy.position}
                  </span>
                </div>
                <div className="flex justify-between items-center bg-slate-950 p-3 rounded-lg">
                  <span className="text-slate-400">Strategy Equity</span>
                  <span className="font-mono text-slate-200">{strategy.equity_value?.toFixed(2)}x</span>
                </div>
                <div className="text-xs text-slate-500 mt-2 flex items-center gap-1">
                  <Clock className="w-3 h-3" /> Last calc: {strategy.time}
                </div>
              </div>
            ) : (
              <div className="text-slate-500 italic">Signal currently unavailable</div>
            )}
          </div>
        </div>

        {/* EWI Signal */}
        <div className="bg-slate-900 p-6 rounded-xl border border-slate-800 shadow-lg flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-semibold text-slate-200">Early Warning Indicator (Paper 10)</h3>
              <AlertTriangle className="text-amber-400 w-5 h-5" />
            </div>
            <p className="text-sm text-slate-400 mb-4">High Volatility prediction model</p>

            {ewi?.status === 'ok' ? (
              <div className="space-y-2">
                <div className="flex justify-between items-center bg-slate-950 p-3 rounded-lg">
                  <span className="text-slate-400">Alert Status</span>
                  <span className={`font-bold flex items-center gap-1 ${ewi.alert_triggered ? 'text-red-400' : 'text-emerald-400'}`}>
                    {ewi.alert_triggered ? (
                      <><AlertTriangle className="w-4 h-4" /> Triggered</>
                    ) : (
                      <><CheckCircle className="w-4 h-4" /> Safe</>
                    )}
                  </span>
                </div>
                <div className="flex justify-between items-center bg-slate-950 p-3 rounded-lg">
                  <span className="text-slate-400">Probability (Score)</span>
                  <span className="font-mono text-slate-200">{(ewi.ewi_probability * 100).toFixed(1)}%</span>
                </div>
                <div className="text-xs text-slate-500 mt-2 flex items-center gap-1">
                  <Clock className="w-3 h-3" /> Last calc: {ewi.time}
                </div>
              </div>
            ) : (
              <div className="text-slate-500 italic">Signal currently unavailable</div>
            )}
          </div>
        </div>

      </div>
    </section>
  );
}
