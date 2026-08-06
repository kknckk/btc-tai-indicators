'use client';

import React from 'react';
import useSWR from 'swr';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, ScatterChart, Scatter, ReferenceLine
} from 'recharts';
import { ShieldAlert, CheckCircle, Activity, Layers, TrendingUp, AlertTriangle } from 'lucide-react';

const fetcher = (url: string) => fetch(url).then(r => r.json());

export default function Paper10Dashboard() {
  const { data, error, isLoading } = useSWR('/api/v1/literature/paper/paper10', fetcher);

  if (isLoading) return <div className="animate-pulse p-8 bg-slate-900 rounded-xl text-slate-400">Ładowanie Systemu Wczesnego Ostrzegania V2...</div>;
  if (error || !data) return <div className="text-red-400 p-8 bg-slate-900 rounded-xl border border-red-900/50">Błąd podczas ładowania danych dla Paper 10.</div>;

  const eventStudy = data.event_study_v2 || {};
  const vsRandom = eventStudy.vs_random || {};
  const timeSeries = data.time_series || [];
  
  // Format data for Event Study Bar Chart
  const eventStudyData = [
    { name: '7D', alert: (eventStudy.avg_return_7d || 0) * 100, random: (vsRandom.random_avg_return_7d || 0) * 100 },
    { name: '14D', alert: (eventStudy.avg_return_14d || 0) * 100, random: (vsRandom.random_avg_return_14d || 0) * 100 },
    { name: '30D', alert: (eventStudy.avg_return_30d || 0) * 100, random: (vsRandom.random_avg_return_30d || 0) * 100 },
  ];

  return (
    <div className="space-y-8">
      {/* Opis metodologiczny */}
      <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-2xl shadow-xl backdrop-blur-sm">
        <div className="flex items-start gap-3">
          <ShieldAlert className="w-6 h-6 text-red-400 mt-1 flex-shrink-0" />
          <div>
            <h3 className="font-bold text-lg text-white mb-2">Skalibrowany System Wczesnego Ostrzegania (EWI V2)</h3>
            <p className="text-slate-300 text-sm leading-relaxed">
              Konstrukcja wieloczynnikowego wskaźnika ryzyka wykorzystująca modele Machine Learning (np. <span className="text-blue-400 font-semibold">HistGradientBoosting</span>). Model jest <span className="font-mono text-xs">skalibrowany izotonicznie</span> i uczy się przewidywać wejście w reżim 5% najwyższej zmienności rozszerzającym się oknem bez Look-Ahead Biasu. Następnie wyznacza próg dla alarmów, optymalizując wskaźnik Precyzji i Recallu dla inwestorów.
            </p>
          </div>
        </div>
      </div>

      {/* 1. Kafelki metryk wczesnego ostrzegania i Event Study */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Metryki EWI */}
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg flex flex-col justify-between">
          <div>
            <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">Alert Quality Score</span>
            <div className="text-4xl font-bold font-mono text-white mt-2">
              {data.alert_quality_score?.toFixed(3)} 
            </div>
            <p className="text-slate-400 text-xs mt-2 mb-4">Miernik asymetrycznych zwrotów wyzwalanych przez system względem losowych inwestycji.</p>
            
            <div className="grid grid-cols-2 gap-3 mt-4 text-xs text-slate-300 border-t border-slate-800 pt-3">
              <div>Alerty OOS: <span className="font-mono font-bold text-amber-400">{eventStudy.alerts_count_in_test}</span></div>
              <div>Częstotliwość: <span className="font-mono font-bold text-amber-400">{eventStudy.alerts_per_year?.toFixed(1)} / rok</span></div>
            </div>
            
            <div className="mt-4 pt-3 border-t border-slate-800/60 text-xs text-slate-400 space-y-2">
              <div className="flex justify-between">
                <span>Model:</span>
                <span className="text-slate-200">{data.best_model}</span>
              </div>
              <div className="flex justify-between">
                <span>Kalibracja:</span>
                <span className="text-slate-200">Isotonic (Expanding window)</span>
              </div>
              <div className="flex justify-between">
                <span>Max F1 Threshold:</span>
                <span className="text-slate-200">{data.threshold_optimization?.max_f1_threshold?.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Wykres Event Study */}
        <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg lg:col-span-2 flex flex-col justify-between">
          <div>
            <h3 className="font-bold text-lg text-blue-400 mb-3 flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Event Study: Zwroty Oczekiwane Po Alercie (vs Baseline)
            </h3>
            <p className="text-slate-400 text-xs mb-4">Przeciętne zakumulowane stopy zwrotu po zapaleniu się alertu w porównaniu z losowym wyborem dnia transakcyjnego ("Monte Carlo Baseline").</p>
            <div className="h-[220px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={eventStudyData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" stroke="#64748b" />
                  <YAxis stroke="#94a3b8" tickFormatter={(v) => `${v}%`} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} formatter={(v: unknown) => [`${Number(v).toFixed(2)}%`, 'Zwrot']} />
                  <Bar dataKey="alert" name="Po Alercie EWI" fill="#10b981" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="random" name="Losowy Wybór (Baseline)" fill="#64748b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Wykres Prawdopodobieństwa EWI vs Zmienność */}
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-lg">
        <h3 className="font-bold text-xl text-red-400 mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5" />
          Skalibrowane Prawdopodobieństwo Alarmowe a Zmienność 30-dniowa
        </h3>
        <p className="text-slate-400 text-sm mb-6">Linia czerwona przedstawia wygenerowane prawdopodobieństwo wejścia w kryzys. Kropki wskazują uruchomienie powiadomienia (Alert Trigger).</p>
        <div className="h-[400px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={timeSeries} margin={{ top: 10, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time" stroke="#64748b" tickFormatter={(val) => val.split('-')[0]} minTickGap={30} />
              <YAxis yAxisId="left" stroke="#3b82f6" domain={['auto', 'auto']} tickFormatter={(val) => val.toFixed(2)} />
              <YAxis yAxisId="right" orientation="right" stroke="#ef4444" domain={[0, 1]} tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} />
              
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }} 
                formatter={(val: unknown, name: any) => {
                  if (name === 'ewi') return [`${(Number(val)*100).toFixed(1)}%`, 'Prawdopodobieństwo EWI'];
                  if (name === 'rv_30d') return [Number(val).toFixed(4), 'Zmienność 30d'];
                  return [val, name];
                }} 
              />
              
              <ReferenceLine yAxisId="right" y={data.threshold_optimization?.min_prec25_threshold || 0.5} stroke="#ef4444" strokeDasharray="4 4" label={{ value: 'Próg', fill: '#ef4444', position: 'insideTopRight' }} />
              
              <Line yAxisId="left" type="monotone" dataKey="rv_30d" stroke="#38bdf8" dot={false} strokeWidth={2} name="rv_30d" />
              <Line yAxisId="right" type="monotone" dataKey="ewi" stroke="#ef4444" dot={false} strokeWidth={2} name="ewi" opacity={0.8} />
              
              {/* Scatter for alerts */}
              <Line yAxisId="right" type="monotone" dataKey={(d: any) => d.alert_prec25 ? d.ewi : null} stroke="none" dot={{ r: 4, fill: '#f59e0b', strokeWidth: 0 }} name="Alert Trigger" isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
