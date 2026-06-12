"use client";
import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, PriceScaleMode, LineSeries } from 'lightweight-charts';

export interface SeriesData {
  time: string;
  value: number;
}

export interface ChartSeries {
  id: string;
  name: string;
  color: string;
  data: SeriesData[];
  axis: 'left' | 'right';
  logarithmic?: boolean;
}

interface ChartProps {
  series: ChartSeries[];
  title: string;
}

export default function Chart({ series, title }: ChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

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
      height: 500,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const seriesRefs: any[] = [];

    series.forEach((s) => {
      // Configure price scale for the axis
      chart.priceScale(s.axis).applyOptions({
        visible: true,
        mode: s.logarithmic ? PriceScaleMode.Logarithmic : PriceScaleMode.Normal,
      });

      const lineSeries = chart.addSeries(LineSeries, {
        color: s.color,
        lineWidth: 2,
        priceScaleId: s.axis,
        title: s.name,
      });
      
      const formattedData = s.data.map(d => ({
        time: d.time as any,
        value: d.value
      }));

      // Sort chronological
      formattedData.sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());

      if(formattedData.length > 0) {
        lineSeries.setData(formattedData);
      }

      seriesRefs.push(lineSeries);
    });

    chartRef.current = chart;

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
  }, [series]);

  return (
    <div className="w-full h-full bg-slate-900 rounded-xl p-4 border border-slate-800 shadow-xl">
      <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
        {title}
      </h2>
      <div ref={chartContainerRef} className="w-full h-[500px]" />
    </div>
  );
}
