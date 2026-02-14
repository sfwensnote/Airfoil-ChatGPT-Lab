'use client';

import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    Brush,
} from 'recharts';
import { AirfoilHistory } from '@/types';

interface HistoryChartProps {
    data: AirfoilHistory[];
    className?: string;
}

const darkAxisStyle = { fontSize: 11, fill: '#94a3b8' };

export function HistoryChart({ data, className = '' }: HistoryChartProps) {
    if (data.length === 0) {
        return null;
    }

    // Sort by timestamp ascending and map to chart data
    const sorted = [...data].sort(
        (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    const chartData = sorted.map((item, idx) => ({
        idx: idx + 1,
        nacaCode: item.nacaCode,
        ldMax: item.ldMax ? parseFloat(item.ldMax.toFixed(2)) : 0,
        cl: item.cl ? parseFloat(item.cl.toFixed(4)) : 0,
        cd: item.cd ? parseFloat((item.cd * 1000).toFixed(2)) : 0, // Scale CD ×1000 for readability
        alpha: item.alpha,
        timestamp: new Date(item.timestamp).toLocaleString(),
    }));

    // Show last N points by default in the Brush window
    const defaultWindow = 15;
    const startIdx = Math.max(0, chartData.length - defaultWindow);
    const endIdx = chartData.length - 1;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const CustomTooltip = ({ active, payload }: any) => {
        if (!active || !payload?.length) return null;
        const d = payload[0]?.payload;
        return (
            <div className="bg-[hsl(220,20%,14%)] border border-[hsl(220,15%,25%)] rounded-lg px-3 py-2 shadow-xl text-xs space-y-1">
                <p className="text-white font-semibold">NACA {d?.nacaCode}</p>
                <p className="text-slate-400">#{d?.idx} · α = {d?.alpha}°</p>
                {payload.map((entry: { name: string; value: number; color: string }, i: number) => (
                    <p key={i} style={{ color: entry.color }}>
                        {entry.name}: {entry.value}
                    </p>
                ))}
                <p className="text-slate-500 text-[10px]">{d?.timestamp}</p>
            </div>
        );
    };

    return (
        <div className={className}>
            <div style={{ width: '100%', height: 280 }}>
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                        <XAxis
                            dataKey="idx"
                            tick={darkAxisStyle}
                            label={{ value: 'Iteration', position: 'insideBottomRight', offset: -5, fill: '#94a3b8', fontSize: 11 }}
                        />
                        <YAxis
                            yAxisId="left"
                            tick={darkAxisStyle}
                            label={{ value: 'L/D Max', angle: -90, position: 'insideLeft', offset: 10, fill: '#94a3b8', fontSize: 11 }}
                        />
                        <YAxis
                            yAxisId="right"
                            orientation="right"
                            tick={darkAxisStyle}
                            label={{ value: 'CL / CD×10³', angle: 90, position: 'insideRight', offset: 10, fill: '#94a3b8', fontSize: 11 }}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <Legend
                            wrapperStyle={{ fontSize: 11, color: '#94a3b8' }}
                        />
                        <Line
                            yAxisId="left"
                            type="monotone"
                            dataKey="ldMax"
                            stroke="#34d399"
                            strokeWidth={2}
                            dot={{ fill: '#34d399', r: 3 }}
                            activeDot={{ r: 5 }}
                            name="L/D Max"
                        />
                        <Line
                            yAxisId="right"
                            type="monotone"
                            dataKey="cl"
                            stroke="#60a5fa"
                            strokeWidth={2}
                            dot={{ fill: '#60a5fa', r: 3 }}
                            activeDot={{ r: 5 }}
                            name="CL"
                        />
                        <Line
                            yAxisId="right"
                            type="monotone"
                            dataKey="cd"
                            stroke="#f87171"
                            strokeWidth={2}
                            dot={{ fill: '#f87171', r: 3 }}
                            activeDot={{ r: 5 }}
                            name="CD ×10³"
                        />
                        {chartData.length > defaultWindow && (
                            <Brush
                                dataKey="idx"
                                height={24}
                                stroke="#475569"
                                fill="hsl(220,20%,12%)"
                                travellerWidth={8}
                                startIndex={startIdx}
                                endIndex={endIdx}
                                tickFormatter={() => ''}
                            />
                        )}
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}

export default HistoryChart;
