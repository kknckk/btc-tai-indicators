import useSWR from 'swr';

const fetcher = async (url: string) => {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error('API Error');
  }
  return res.json();
};

export interface DataPoint {
  time: string;
  value: number;
}

export interface IndicatorResponse {
  indicator: string;
  data: DataPoint[];
}

export function useIndicators(metrics: string[], startDate: string = "2010-01-01", endDate: string = "2030-01-01") {
  const queryParams = new URLSearchParams();
  metrics.forEach(m => queryParams.append('metrics', m));
  queryParams.append('start_date', startDate);
  queryParams.append('end_date', endDate);
  
  // Wykorzystujemy adres publiczny API na Cloud Run lub lokalny w razie zmiennej srodowiskowej
  const baseUrl = process.env.NEXT_PUBLIC_API_URL || "https://btc-api-203925818774.europe-central2.run.app";
  const url = `${baseUrl}/api/v1/metrics/values?${queryParams.toString()}`;
  
  const { data, error, isLoading } = useSWR<IndicatorResponse[]>(url, fetcher);
  
  return {
    data,
    isLoading,
    isError: error
  };
}

export interface AvailableIndicatorsResponse {
  indicators: string[];
}

export function useAvailableIndicators() {
  const baseUrl = process.env.NEXT_PUBLIC_API_URL || "https://btc-api-203925818774.europe-central2.run.app";
  const url = `${baseUrl}/api/v1/metrics/available`;
  
  const { data, error, isLoading } = useSWR<AvailableIndicatorsResponse>(url, fetcher);
  
  return {
    data,
    isLoading,
    isError: error
  };
}
