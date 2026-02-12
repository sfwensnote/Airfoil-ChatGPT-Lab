'use client';

import { useEffect, useCallback } from 'react';
import { X, MessageSquare } from 'lucide-react';
import { ChatWindow } from './ChatWindow';

interface ChatDrawerProps {
    isOpen: boolean;
    onClose: () => void;
}

export function ChatDrawer({ isOpen, onClose }: ChatDrawerProps) {
    // ESC key to close
    const handleKeyDown = useCallback((e: KeyboardEvent) => {
        if (e.key === 'Escape') onClose();
    }, [onClose]);

    useEffect(() => {
        if (isOpen) {
            document.addEventListener('keydown', handleKeyDown);
            document.body.style.overflow = 'hidden';
        }
        return () => {
            document.removeEventListener('keydown', handleKeyDown);
            document.body.style.overflow = '';
        };
    }, [isOpen, handleKeyDown]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex justify-end">
            {/* Backdrop overlay */}
            <div
                className="absolute inset-0 bg-black/50 backdrop-blur-sm drawer-overlay"
                onClick={onClose}
            />

            {/* Drawer panel */}
            <div className="relative w-[420px] max-w-[90vw] h-full drawer-panel flex flex-col">
                {/* Drawer header */}
                <div className="flex items-center justify-between px-4 py-3 border-b border-[hsl(220,15%,22%)] bg-[hsl(220,20%,12%)] flex-shrink-0">
                    <div className="flex items-center gap-2">
                        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                            <MessageSquare className="w-4 h-4 text-white" />
                        </div>
                        <h2 className="text-sm font-semibold text-white">AI Tutor</h2>
                    </div>
                    <button
                        onClick={onClose}
                        className="w-7 h-7 rounded-lg bg-[hsl(220,15%,20%)] hover:bg-[hsl(220,15%,25%)] flex items-center justify-center transition-colors"
                    >
                        <X className="w-4 h-4 text-slate-400" />
                    </button>
                </div>

                {/* Chat content */}
                <div className="flex-1 overflow-hidden bg-[hsl(220,20%,10%)]">
                    <ChatWindow className="h-full border-0 shadow-none rounded-none bg-transparent" />
                </div>
            </div>
        </div>
    );
}

export default ChatDrawer;
