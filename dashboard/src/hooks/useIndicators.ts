import useSWR from 'swr';

export const getIndicatorGroup = (name: string): string => {
  const mockowane = ["SplyAct30d", "SplyAct180d", "SplyAct1yr", "CapAct1yrUSD"];
  const darmowe_api = [
    "CapMrktCurUSD", "CapMVRVCur", "FeeTotNtv", "AdrActCnt", "AdrBalCnt", 
    "TxCnt", "TxTfrCnt", "BlkCnt", "HashRate", "SplyCur", "IssTotNtv", "IssTotUSD", 
    "SplyExpFut10yr", "FlowInExNtv", "FlowInExUSD", "FlowOutExNtv", "FlowOutExUSD", 
    "PriceBTC", "PriceUSD", "TxTfrValAdjUSD", "TxTfrValAdjNtv", "TxTfrValNtv", 
    "TxTfrValUSD", "BlkSizeMeanByte", "UTXOCnt", "SOPR", "CDD", "LTH_MVRV", "STH_MVRV", "FeeMedNtv"
  ];
  if (mockowane.includes(name)) return "3. Wskaźniki symulowane (Mockowane)";
  if (darmowe_api.includes(name)) return "1. Pobierane bezpośrednio z darmowych API";
  return "2. Wyliczane u nas (Derywaty i agregacje)";
};

export const GROUP_ORDER = [
  "1. Pobierane bezpośrednio z darmowych API",
  "2. Wyliczane u nas (Derywaty i agregacje)",
  "3. Wskaźniki symulowane (Mockowane)",
];

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

export interface CorrelationResponse {
  indicator: string;
  correlation: number;
}

export function useCorrelation(days: number = 90, targetMetric: string = "PriceUSD") {
  const baseUrl = process.env.NEXT_PUBLIC_API_URL || "https://btc-api-203925818774.europe-central2.run.app";
  const url = `${baseUrl}/api/v1/metrics/correlation?days=${days}&target_metric=${targetMetric}`;
  
  const { data, error, isLoading } = useSWR<CorrelationResponse[]>(url, fetcher);
  
  return {
    data,
    isLoading,
    isError: error
  };
}
