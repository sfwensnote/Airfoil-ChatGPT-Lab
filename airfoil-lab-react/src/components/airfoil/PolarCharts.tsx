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
    ReferenceLine,
} from 'recharts';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { PolarPoint } from '@/types';

interface PolarChartsProps {
    data: PolarPoint[];
    currentAlpha: number;
    optimalAlpha?: number;
    className?: string;
}

const darkAxisStyle = { fontSize: 11, fill: '#94a3b8' };
const darkLabelStyle = { fill: '#94a3b8', fontSize: 12 };

export function PolarCharts({
    data,
    currentAlpha,
    optimalAlpha,
    className = '',
}: PolarChartsProps) {
    const dataWithLD = data.map((point) => ({
        ...point,
        LD: point.CD > 1e-12 ? point.CL / point.CD : 0,
    }));

    const currentPoint = data.find(
        (d) => Math.abs(d.alpha - currentAlpha) < 0.1
    );

    return (
        <div className={className}>
            <Tabs defaultValue="cl" className="w-full">
                <TabsList className="grid w-full grid-cols-3 mb-4 bg-white/5 border border-white/5">
                    <TabsTrigger value="cl" className="text-sm data-[state=active]:bg-white/10 data-[state=active]:text-white text-slate-400 cursor-pointer">CL vs α</TabsTrigger>
                    <TabsTrigger value="polar" className="text-sm data-[state=active]:bg-white/10 data-[state=active]:text-white text-slate-400 cursor-pointer">CL vs CD</TabsTrigger>
                    <TabsTrigger value="ld" className="text-sm data-[state=active]:bg-white/10 data-[state=active]:text-white text-slate-400 cursor-pointer">L/D vs α</TabsTrigger>
                </TabsList>

                {/* CL vs Alpha */}
                <TabsContent value="cl" className="mt-0">
                    <div className="h-64 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                                <XAxis
                                    dataKey="alpha"
                                    label={{ value: 'α (°)', position: 'insideBottom', offset: -5, ...darkLabelStyle }}
                                    tick={darkAxisStyle}
                                    stroke="rgba(148,163,184,0.2)"
                                />
                                <YAxis
                                    label={{ value: 'CL', angle: -90, position: 'insideLeft', ...darkLabelStyle }}
                                    tick={darkAxisStyle}
                                    stroke="rgba(148,163,184,0.2)"
                                />
                                <Tooltip
                                    formatter={(value) => typeof value === 'number' ? value.toFixed(4) : value}
                                    labelFormatter={(label) => `α = ${label}°`}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="CL"
                                    stroke="#60a5fa"
                                    strokeWidth={2}
                                    dot={{ r: 2, fill: '#60a5fa' }}
                                    activeDot={{ r: 5, fill: '#3b82f6', stroke: '#60a5fa', strokeWidth: 2 }}
                                />
                                <ReferenceLine
                                    x={currentAlpha}
                                    stroke="#f87171"
                                    strokeDasharray="5 5"
                                    label={{ value: `α=${currentAlpha}°`, position: 'top', fontSize: 10, fill: '#f87171' }}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </TabsContent>

                {/* Drag Polar (CL vs CD) */}
                <TabsContent value="polar" className="mt-0">
                    <div className="h-64 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                                <XAxis
                                    dataKey="CD"
                                    label={{ value: 'CD', position: 'insideBottom', offset: -5, ...darkLabelStyle }}
                                    tick={darkAxisStyle}
                                    tickFormatter={(v) => v.toFixed(4)}
                                    stroke="rgba(148,163,184,0.2)"
                                />
                                <YAxis
                                    dataKey="CL"
                                    label={{ value: 'CL', angle: -90, position: 'insideLeft', ...darkLabelStyle }}
                                    tick={darkAxisStyle}
                                    stroke="rgba(148,163,184,0.2)"
                                />
                                <Tooltip
                                    formatter={(value, name) => [typeof value === 'number' ? value.toFixed(4) : value, name]}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="CL"
                                    stroke="#a78bfa"
                                    strokeWidth={2}
                                    dot={{ r: 2, fill: '#a78bfa' }}
                                    activeDot={{ r: 5, fill: '#8b5cf6', stroke: '#a78bfa', strokeWidth: 2 }}
                                />
                                {currentPoint && (
                                    <ReferenceLine
                                        x={currentPoint.CD}
                                        stroke="#f87171"
                                        strokeDasharray="5 5"
                                    />
                                )}
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </TabsContent>

                {/* L/D vs Alpha */}
                <TabsContent value="ld" className="mt-0">
                    <div className="h-64 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={dataWithLD} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                                <XAxis
                                    dataKey="alpha"
                                    label={{ value: 'α (°)', position: 'insideBottom', offset: -5, ...darkLabelStyle }}
                                    tick={darkAxisStyle}
                                    stroke="rgba(148,163,184,0.2)"
                                />
                                <YAxis
                                    label={{ value: 'L/D', angle: -90, position: 'insideLeft', ...darkLabelStyle }}
                                    tick={darkAxisStyle}
                                    stroke="rgba(148,163,184,0.2)"
                                />
                                <Tooltip
                                    formatter={(value) => typeof value === 'number' ? value.toFixed(2) : value}
                                    labelFormatter={(label) => `α = ${label}°`}
                                />
                                <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
                                <Line
                                    type="monotone"
                                    dataKey="LD"
                                    name="L/D"
                                    stroke="#34d399"
                                    strokeWidth={2}
                                    dot={{ r: 2, fill: '#34d399' }}
                                    activeDot={{ r: 5, fill: '#10b981', stroke: '#34d399', strokeWidth: 2 }}
                                />
                                <ReferenceLine
                                    x={currentAlpha}
                                    stroke="#f87171"
                                    strokeDasharray="5 5"
                                />
                                {optimalAlpha !== undefined && (
                                    <ReferenceLine
                                        x={optimalAlpha}
                                        stroke="#34d399"
                                        strokeDasharray="3 3"
                                        label={{ value: 'Optimal', position: 'top', fontSize: 10, fill: '#34d399' }}
                                    />
                                )}
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </TabsContent>
            </Tabs>
        </div>
    );
}

export default PolarCharts;
