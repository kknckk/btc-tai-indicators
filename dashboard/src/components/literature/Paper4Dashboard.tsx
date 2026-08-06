'use client';

import React from 'react';
import useSWR from 'swr';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, AreaChart, Area, Scatter
} from 'recharts';
import { Award, TrendingUp, ShieldAlert, Zap, BarChart3, Layers } from 'lucide-react';

const fetcher = (url: string) => fetch(url).then(r => r.json());

export default function Paper4Dashboard() {
  const { data, error, isLoading } = useSWR('/api/v1/literature/paper/paper4', fetcher);

  if (isLoading) return <div className="animate-pulse p-8 bg-slate-900 rounded-xl text-slate-400">Ładowanie wyników strategii inwestycyjnych V2 (Zero Look-Ahead)...</div>;
  if (error || !data) return <div className="text-red-400 p-8 bg-slate-900 rounded-xl border border-red-900/50">Błąd podczas ładowania danych dla Paper 4.</div>;

  const strat = data.expanding?.metrics || {};
  const bh = data.buy_and_hold?.metrics || {};
  const time_series = data.expanding?.equity_curve || [];

  return (
    <div className="space-y-8">
      {/* Opis metodologiczny */}
      <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-2xl shadow-xl backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <Layers className="w-6 h-6 text-amber-400 mt-1 flex-shrink-0" />
          <div>
            <h3 className="font-bold text-lg text-white mb-2">Backtesting Strategii "Expanding Window" (Bez Look-Ahead Bias)</h3>
            <p className="text-slate-300 text-sm leading-relaxed">
              Symulacja inwestycyjna wykorzystująca rozszerzające się okno historyczne (<span className="text-amber-400 font-semibold">Expanding Window</span>). Strategia dynamicznie oblicza kwantyle on-chain (np. <span className="text-emerald-400 font-semibold">MVRV Z-Score</span>) bazując WYŁĄCZNIE na danych dostępnych do danego dnia. Generuje sygnały alokacji chroniąc przed Max Drawdownem. Koszty transakcyjne wliczone (20 bps).
            </p>
          </div>
        </div>
      </div>

      {/* 1. Kafelki porównawcze: Strategia vs Buy & Hold */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl relative overflow-hidden shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Sharpe Ratio</span>
            <Award className="w-5 h-5 text-amber-400" />
          </div>
          <div className="text-3xl font-bold font-mono text-white mb-1">{strat.sharpe?.toFixed(2) ?? '0.00'}</div>
          <div className="text-xs text-slate-400 flex items-center justify-between">
            <span>Buy & Hold: <span className="font-mono text-slate-300">{bh.sharpe?.toFixed(2) ?? '0.00'}</span></span>
            <span className={`font-semibold ${strat.sharpe >= bh.sharpe ? 'text-emerald-400' : 'text-red-400'}`}>
              {strat.sharpe >= bh.sharpe ? '+' : ''}{((strat.sharpe - bh.sharpe)).toFixed(2)}
            </span>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl relative overflow-hidden shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Maksymalne Osunięcie (Max DD)</span>
            <ShieldAlert className="w-5 h-5 text-red-400" />
          </div>
          <div className="text-3xl font-bold font-mono text-emerald-400 mb-1">{((strat.max_drawdown ?? 0) * 100).toFixed(1)}%</div>
          <div className="text-xs text-slate-400 flex items-center justify-between">
            <span>Buy & Hold: <span className="font-mono text-slate-300">{((bh.max_drawdown ?? 0) * 100).toFixed(1)}%</span></span>
            <span className="text-emerald-400 font-semibold">Ochrona kapitału</span>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl relative overflow-hidden shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Sortino Ratio</span>
            <TrendingUp className="w-5 h-5 text-blue-400" />
          </div>
          <div className="text-3xl font-bold font-mono text-white mb-1">{strat.sortino?.toFixed(2) ?? '0.00'}</div>
          <div className="text-xs text-slate-400 flex items-center justify-between">
            <span>Buy & Hold: <span className="font-mono text-slate-300">{bh.sortino?.toFixed(2) ?? '0.00'}</span></span>
            <span className={`font-semibold ${strat.sortino >= bh.sortino ? 'text-emerald-400' : 'text-red-400'}`}>
              {(strat.sortino / (bh.sortino || 1)).toFixed(1)}x wyższe
            </span>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl relative overflow-hidden shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Liczba Transakcji</span>
            <Zap className="w-5 h-5 text-purple-400" />
          </div>
          <div className="text-3xl font-bold font-mono text-purple-400 mb-1">{strat.transactions ?? 0}</div>
          <div className="text-xs text-slate-400">
            <span>Ekspozycja rynkowa: <span className="text-slate-300 font-semibold">{strat.pct_time_in_market?.toFixed(1)}% czasu</span></span>
          </div>
        </div>
      </div>

      {/* 2. Wykres Krzywej Kapitału (Equity Curve) */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-emerald-400 mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          Krzywa Kapitału: Expanding Strategy vs Buy & Hold (Log Scale)
        </h3>
        <p className="text-slate-400 text-sm mb-6">Rozwój portfela przy inwestycji początkowej równej 1 (skala logarytmiczna zlicza wartość kapitału). Szare kropki pod wykresem oznaczają dni kiedy strategia była "w rynku" (Long).</p>
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={time_series} margin={{ top: 10, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time" stroke="#64748b" tickFormatter={(val) => val.split('-')[0]} minTickGap={30} />
              
              <YAxis yAxisId="left" stroke="#10b981" scale="log" domain={['auto', 'auto']} tickFormatter={(val) => `${val.toFixed(1)}x`} />
              
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} 
                formatter={(val: unknown, name: any) => {
                  if (name === 'strategy_value') return [`${Number(val).toFixed(2)}x`, 'Kapitał Strategii'];
                  if (name === 'bh_value') return [`${Number(val).toFixed(2)}x`, 'Buy & Hold'];
                  if (name === 'MVRV_Z') return [Number(val).toFixed(2), 'MVRV Z-Score'];
                  return [val, name];
                }} 
              />
              
              <Line yAxisId="left" type="monotone" dataKey="bh_value" stroke="#64748b" dot={false} strokeWidth={2} name="bh_value" opacity={0.6} />
              <Line yAxisId="left" type="monotone" dataKey="strategy_value" stroke="#10b981" dot={false} strokeWidth={2} name="strategy_value" />
              
              {/* Sygnały Long */}
              <Scatter yAxisId="left" dataKey={(d: any) => d.position === 1 ? 0.8 : null} fill="#38bdf8" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}
