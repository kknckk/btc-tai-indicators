"use client";
import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi } from 'lightweight-charts';

interface ChartProps {
  data: { time: string; value: number }[];
  title: string;
}

export default function Chart({ data, title }: ChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: 'rgba(42, 46, 57, 0.5)' },
        horzLines: { color: 'rgba(42, 46, 57, 0.5)' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 400,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const series = chart.addLineSeries({
      color: '#3b82f6',
      lineWidth: 2,
    });
    
    // Konwersja czasu na timestamp lub format 'yyyy-mm-dd'
    const formattedData = data.map(d => ({
      time: d.time as any,
      value: d.value
    }));

    // Posortuj dane rosnąco po czasie
    formattedData.sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());

    if(formattedData.length > 0) {
      series.setData(formattedData);
    }

    chartRef.current = chart;
    seriesRef.current = series;

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
      }
    };
  }, [data]);

  return (
    <div className="w-full h-full bg-slate-900 rounded-xl p-4 border border-slate-800 shadow-xl">
      <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
        {title}
      </h2>
      <div ref={chartContainerRef} className="w-full h-[400px]" />
    </div>
  );
}
