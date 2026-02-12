'use client';

import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import { useState, useEffect } from 'react';

interface ParameterSliderProps {
    label: string;
    value: number;
    min: number;
    max: number;
    step: number;
    unit?: string;
    onChange: (value: number) => void;
    formatValue?: (value: number) => string;
    help?: string;
    className?: string;
}

export function ParameterSlider({
    label,
    value,
    min,
    max,
    step,
    unit = '',
    onChange,
    formatValue,
    help,
    className = '',
}: ParameterSliderProps) {
    const [inputValue, setInputValue] = useState(value.toString());

    useEffect(() => {
        setInputValue(formatValue ? formatValue(value) : value.toFixed(2));
    }, [value, formatValue]);

    const handleSliderChange = (values: number[]) => {
        const newValue = values[0];
        onChange(newValue);
        setInputValue(formatValue ? formatValue(newValue) : newValue.toFixed(2));
    };

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setInputValue(e.target.value);
    };

    const handleInputBlur = () => {
        let newValue = parseFloat(inputValue);
        if (isNaN(newValue)) {
            newValue = value;
        } else {
            newValue = Math.max(min, Math.min(max, newValue));
            newValue = Math.round((newValue - min) / step) * step + min;
        }
        onChange(newValue);
        setInputValue(formatValue ? formatValue(newValue) : newValue.toFixed(2));
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            handleInputBlur();
        }
    };

    return (
        <div className={`space-y-2 ${className}`}>
            <div className="flex justify-between items-center">
                <label className="text-sm font-medium text-slate-300">
                    {label}
                    {help && (
                        <span className="ml-1 text-slate-500 cursor-help" title={help}>
                            ⓘ
                        </span>
                    )}
                </label>
                <span className="text-sm text-slate-400 font-mono">
                    {formatValue ? formatValue(value) : value.toFixed(2)}{unit}
                </span>
            </div>
            <div className="flex gap-3 items-center">
                <div className="flex-1">
                    <Slider
                        value={[value]}
                        min={min}
                        max={max}
                        step={step}
                        onValueChange={handleSliderChange}
                        className="w-full"
                    />
                </div>
                <div className="w-20">
                    <Input
                        type="text"
                        value={inputValue}
                        onChange={handleInputChange}
                        onBlur={handleInputBlur}
                        onKeyDown={handleKeyDown}
                        className="h-8 text-center text-sm bg-white/5 border-white/10 text-slate-200 focus:border-blue-500/50"
                    />
                </div>
            </div>
        </div>
    );
}

export default ParameterSlider;
