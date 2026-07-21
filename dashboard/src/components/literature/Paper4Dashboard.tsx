'use client';

import React from 'react';
import useSWR from 'swr';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, AreaChart, Area
} from 'recharts';
import { Award, TrendingUp, ShieldAlert, Zap, BarChart3, Layers } from 'lucide-react';

const fetcher = (url: string) => fetch(url).then(r => r.json());

export default function Paper4Dashboard() {
  const { data, error, isLoading } = useSWR('/api/v1/literature/paper/paper4', fetcher);

  if (isLoading) return <div className="animate-pulse p-8 bg-slate-900 rounded-xl text-slate-400">Ładowanie wyników strategii inwestycyjnych (VectorBT)...</div>;
  if (error || !data) return <div className="text-red-400 p-8 bg-slate-900 rounded-xl border border-red-900/50">Błąd podczas ładowania danych dla Paper 4.</div>;

  const { metrics = {}, thresholds = {}, time_series = [] } = data;
  const strat = metrics.strategy || {};
  const bh = metrics.buy_and_hold || {};

  return (
    <div className="space-y-8">
      {/* Opis metodologiczny */}
      <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-2xl shadow-xl backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <Layers className="w-6 h-6 text-amber-400 mt-1 flex-shrink-0" />
          <div>
            <h3 className="font-bold text-lg text-white mb-2">Backtesting Strategii na bazie MVRV Z-Score i NUPL</h3>
            <p className="text-slate-300 text-sm leading-relaxed">
              Kwantylowa strategia inwestycyjna oparta na wskaźnikach on-chain <span className="text-amber-400 font-semibold">MVRV Z-Score</span> (skorygowanego o zrealizowaną kapitalizację) oraz <span className="text-emerald-400 font-semibold">NUPL</span> (Net Unrealized Profit/Loss). Strategia generuje sygnały kupna, gdy rynek wkracza w strefę skrajnego zaniżenia wyceny (<span className="font-mono text-xs text-emerald-400">MVRV &lt; {thresholds.mvrv_low?.toFixed(2)}</span>), oraz sygnały wyjścia przy przegrzaniu (<span className="font-mono text-xs text-red-400">MVRV &gt; {thresholds.mvrv_high?.toFixed(2)}</span>).
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
          <div className="text-3xl font-bold font-mono text-purple-400 mb-1">{strat.trades ?? 0}</div>
          <div className="text-xs text-slate-400">
            <span>Średni hold time: <span className="text-slate-300 font-semibold">Długoterminowy</span></span>
          </div>
        </div>
      </div>

      {/* 2. Wykres MVRV Z-Score z progami wejścia/wyjścia */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-amber-400 mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5" />
          MVRV Z-Score a Strefy Decyzyjne Strategii
        </h3>
        <p className="text-slate-400 text-sm mb-6">Dolna linia przerywana (<span className="text-emerald-400 font-mono font-bold">{thresholds.mvrv_low?.toFixed(2)}</span>) sygnalizuje strefę kupna, górna (<span className="text-red-400 font-mono font-bold">{thresholds.mvrv_high?.toFixed(2)}</span>) strefę realizacji zysków.</p>
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={time_series} margin={{ top: 10, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time" stroke="#64748b" tickFormatter={(val) => val.split('-')[0]} />
              <YAxis yAxisId="left" stroke="#3b82f6" domain={['auto', 'auto']} tickFormatter={(val) => `$${(val/1000).toFixed(0)}k`} />
              <YAxis yAxisId="right" orientation="right" stroke="#f59e0b" domain={[-1, 8]} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} />
              <ReferenceLine yAxisId="right" y={thresholds.mvrv_low ?? 0} stroke="#10b981" strokeDasharray="4 4" label={{ value: 'Kupno', fill: '#10b981', position: 'insideTopRight' }} />
              <ReferenceLine yAxisId="right" y={thresholds.mvrv_high ?? 3.5} stroke="#ef4444" strokeDasharray="4 4" label={{ value: 'Sprzedaż', fill: '#ef4444', position: 'insideBottomRight' }} />
              <Line yAxisId="left" type="monotone" dataKey="PriceUSD" stroke="#3b82f6" dot={false} strokeWidth={2} name="Cena USD" />
              <Line yAxisId="right" type="monotone" dataKey="MVRV_Z" stroke="#f59e0b" dot={false} strokeWidth={1.5} name="MVRV Z-Score" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 3. Wykres NUPL (Net Unrealized Profit/Loss) */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-emerald-400 mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          NUPL (Net Unrealized Profit/Loss)
        </h3>
        <p className="text-slate-400 text-sm mb-6">Dynamika niezrealizowanych zysków i strat w sieci Bitcoin w skali od -0.5 do 1.0.</p>
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={time_series} margin={{ top: 10, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time" stroke="#64748b" tickFormatter={(val) => val.split('-')[0]} />
              <YAxis stroke="#10b981" domain={[-0.5, 1.0]} tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} />
              <ReferenceLine y={0} stroke="#64748b" />
              <Area type="monotone" dataKey="NUPL" stroke="#10b981" fill="#10b981" fillOpacity={0.2} name="NUPL" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
