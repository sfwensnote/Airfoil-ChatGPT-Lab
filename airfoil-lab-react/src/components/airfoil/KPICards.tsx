'use client';

import { Card, CardContent } from '@/components/ui/card';

interface KPIData {
    cl: number;
    cd: number;
    ld: number;
    alphaOpt: number;
    ldMax: number;
}

interface KPICardsProps {
    data: KPIData;
    className?: string;
}

export function KPICards({ data, className = '' }: KPICardsProps) {
    const cards = [
        {
            label: 'Lift (CL)',
            value: data.cl.toFixed(3),
            color: 'text-blue-400',
            borderColor: 'border-blue-500/20',
            glowColor: 'shadow-blue-500/5',
        },
        {
            label: 'Drag (CD)',
            value: data.cd.toFixed(5),
            color: 'text-purple-400',
            borderColor: 'border-purple-500/20',
            glowColor: 'shadow-purple-500/5',
        },
        {
            label: 'Efficiency (L/D)',
            value: data.ld.toFixed(1),
            color: 'text-amber-400',
            borderColor: 'border-amber-500/20',
            glowColor: 'shadow-amber-500/5',
        },
        {
            label: `Max L/D (@${data.alphaOpt.toFixed(1)}°)`,
            value: data.ldMax.toFixed(1),
            color: 'text-emerald-400',
            borderColor: 'border-emerald-500/20',
            glowColor: 'shadow-emerald-500/5',
        },
    ];

    return (
        <div className={`grid grid-cols-2 md:grid-cols-4 gap-3 ${className}`}>
            {cards.map((card, index) => (
                <div
                    key={index}
                    className={`card-panel p-4 text-center border ${card.borderColor} shadow-lg ${card.glowColor}`}
                >
                    <div className="text-[10px] uppercase tracking-wider text-slate-400 mb-1.5 font-medium">
                        {card.label}
                    </div>
                    <div className={`text-2xl font-bold font-mono ${card.color}`}>
                        {card.value}
                    </div>
                </div>
            ))}
        </div>
    );
}

export default KPICards;
