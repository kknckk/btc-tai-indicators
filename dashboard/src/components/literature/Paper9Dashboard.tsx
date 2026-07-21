'use client';

import React from 'react';
import useSWR from 'swr';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, AreaChart, Area
} from 'recharts';
import { Landmark, Activity, Layers, TrendingUp, ShieldAlert } from 'lucide-react';

const fetcher = (url: string) => fetch(url).then(r => r.json());

export default function Paper9Dashboard() {
  const { data, error, isLoading } = useSWR('/api/v1/literature/paper/paper9', fetcher);

  if (isLoading) return <div className="animate-pulse p-8 bg-slate-900 rounded-xl text-slate-400">Ładowanie analiz wolumenu i koncentracji giełd (NBER)...</div>;
  if (error || !data) return <div className="text-red-400 p-8 bg-slate-900 rounded-xl border border-red-900/50">Błąd podczas ładowania danych dla Paper 9.</div>;

  const { stats = {}, time_series = [] } = data;

  return (
    <div className="space-y-8">
      {/* Opis metodologiczny */}
      <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-2xl shadow-xl backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <Landmark className="w-6 h-6 text-sky-400 mt-1 flex-shrink-0" />
          <div>
            <h3 className="font-bold text-lg text-white mb-2">Analiza Koncentracji Wolumenu Giełdowego i Szumu Transakcyjnego (NBER Working Paper)</h3>
            <p className="text-slate-300 text-sm leading-relaxed">
              Badanie weryfikuje hipotezę o relacji rzeczywistego wolumenu gospodarczego rozliczanego na blockchainie z przepływami na giełdach kryptowalutowych (<span className="text-sky-400 font-semibold">ShareExUSD</span>). Udział wolumenu giełdowego w całkowitej wartości transferów on-chain odzwierciedla spekulacyjne nastroje inwestorów oraz filtruje wewnętrzny szum transakcyjny (np. miksery, reorganizacje portfeli giełdowych).
            </p>
          </div>
        </div>
      </div>

      {/* 1. Kafelki statystyczne */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl relative overflow-hidden shadow-lg">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Średni Udział Giełd (Mean Share)</span>
          <div className="text-3xl font-bold font-mono text-white mt-1">{((stats.mean_share ?? 0) * 100).toFixed(1)}%</div>
          <p className="text-xs text-slate-500 mt-2">Średnia część wolumenu on-chain trafiająca na giełdy.</p>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl relative overflow-hidden shadow-lg">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Mediana Udziału (P50)</span>
          <div className="text-3xl font-bold font-mono text-sky-400 mt-1">{((stats.p50_share ?? 0) * 100).toFixed(1)}%</div>
          <p className="text-xs text-slate-500 mt-2">Wartość środkowa eliminująca skrajne szczyty.</p>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl relative overflow-hidden shadow-lg">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">90. Percentyl (Szczyty spekulacji)</span>
          <div className="text-3xl font-bold font-mono text-amber-400 mt-1">{((stats.p90_share ?? 0) * 100).toFixed(1)}%</div>
          <p className="text-xs text-slate-500 mt-2">Dni maksymalnej dominacji handlu giełdowego.</p>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl relative overflow-hidden shadow-lg">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Korelacja z r_t / Zmiennością</span>
          <div className="text-3xl font-bold font-mono text-emerald-400 mt-1">{(stats.corr_vol_rv ?? 0).toFixed(2)}</div>
          <p className="text-xs text-slate-500 mt-2">Korelacja liniowa między ShareExUSD a zrealizowaną zmiennością.</p>
        </div>
      </div>

      {/* 2. Wykres ShareExUSD vs PriceUSD */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-sky-400 mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Szereg Czasowy: ShareExUSD vs Cena Bitcoina
        </h3>
        <p className="text-slate-400 text-sm mb-6">Obserwacja wzrostu aktywności giełd podczas gwałtownych ruchów cenowych.</p>
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={time_series} margin={{ top: 10, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time" stroke="#64748b" tickFormatter={(val) => val.split('-')[0]} />
              <YAxis yAxisId="left" stroke="#3b82f6" domain={['auto', 'auto']} tickFormatter={(val) => `$${(val/1000).toFixed(0)}k`} />
              <YAxis yAxisId="right" orientation="right" stroke="#38bdf8" tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} formatter={(val: any, name: any) => [name === 'ShareExUSD' ? `${(val * 100).toFixed(2)}%` : `$${Number(val).toFixed(2)}`, name]} />
              <ReferenceLine yAxisId="right" y={stats.mean_share ?? 0.15} stroke="#64748b" strokeDasharray="3 3" label={{ value: 'Średnia', fill: '#64748b', position: 'insideTopRight' }} />
              <Line yAxisId="left" type="monotone" dataKey="PriceUSD" stroke="#3b82f6" dot={false} strokeWidth={2} name="Cena USD" />
              <Line yAxisId="right" type="monotone" dataKey="ShareExUSD" stroke="#38bdf8" dot={false} strokeWidth={1.5} name="ShareExUSD" opacity={0.85} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
