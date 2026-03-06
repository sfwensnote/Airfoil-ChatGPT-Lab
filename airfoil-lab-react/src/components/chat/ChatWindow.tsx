'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, BookOpen, FlaskConical, Target, Zap, Search, XCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import { ScrollArea } from '@/components/ui/scroll-area';
import { useChatStore, useSimulationStore, useUserStore } from '@/stores';
import { ChatMessage, GeometryParams } from '@/types';
import { getConversationHistory, saveConversation, chatWithAgent, deleteConversationHistory } from '@/lib/api';
import { v4 as uuidv4 } from 'uuid';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// Toggle for multi-agent system (set to true to use new backend)
const USE_MULTI_AGENT = true;

const AI_MODULES = [
    {
        id: 'Concept Learning' as const,
        label: 'Concept',
        icon: BookOpen,
        color: 'bg-blue-500/15 text-blue-300',
        description: 'Learn aerodynamics concepts'
    },
    {
        id: 'Model Iteration' as const,
        label: 'Iteration',
        icon: FlaskConical,
        color: 'bg-amber-500/15 text-amber-300',
        description: 'Experiment design & analysis'
    },
    {
        id: 'Strategy Review' as const,
        label: 'Strategy',
        icon: Target,
        color: 'bg-purple-500/15 text-purple-300',
        description: 'Review & feedback'
    },
];

export function ChatWindow({ className = '' }: { className?: string }) {
    const [input, setInput] = useState('');
    const [searchTerm, setSearchTerm] = useState('');
    const [roleFilter, setRoleFilter] = useState<'ALL' | string>('ALL');
    const [showSearch, setShowSearch] = useState(false);

    const scrollRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const {
        messages,
        currentModule,
        isStreaming,
        addMessage,
        updateMessage,
        setModule,
        clearMessages,
        setStreaming,
        loadHistory
    } = useChatStore();

    const { geometry, environment, result, setGeometry } = useSimulationStore();
    const { currentUser } = useUserStore();
    const userId = currentUser ? currentUser.username : 'guest';

    // Load history when userId changes
    useEffect(() => {
        const loadChatHistory = async () => {
            if (userId && userId !== 'guest') {
                try {
                    const history = await getConversationHistory(userId);
                    loadHistory(history);
                } catch (error) {
                    console.error('Failed to load chat history:', error);
                }
            } else {
                clearMessages();
            }
        };
        loadChatHistory();
    }, [userId, loadHistory, clearMessages]);

    // Auto scroll to bottom
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    // 插入仿真数据
    const insertSimData = () => {
        const simInfo = result
            ? `Current Simulation:\n- NACA ${result.geometry.nacaCode}\n- α = ${environment.alpha}°\n- CL = ${result.kpi.cl.toFixed(3)}, CD = ${result.kpi.cd.toFixed(5)}\n- L/D = ${result.kpi.ld.toFixed(1)}\n- Re ≈ ${((environment.rho * environment.velocity * environment.chord) / environment.mu).toFixed(0)}`
            : `Current Parameters:\n- Camber: ${geometry.camber}%\n- Thickness: ${geometry.thickness}%\n- α = ${environment.alpha}°\n- No simulation run yet`;

        setInput((prev) => prev + (prev ? '\n\n' : '') + simInfo);
        inputRef.current?.focus();
    };

    // 发送消息
    const handleSend = async () => {
        if (!input.trim() || isStreaming) return;

        const userMessage: ChatMessage = {
            id: uuidv4(),
            role: 'user',
            content: input.trim(),
            timestamp: new Date().toISOString(),
        };

        addMessage(userMessage);
        setInput('');
        setStreaming(true);

        // 计算雷诺数
        const re = (environment.rho * environment.velocity * environment.chord) / environment.mu;

        try {
            let aiContent: string;
            // Default agent (fallback)
            let agentName: string = 'Concept Learning';

            if (USE_MULTI_AGENT) {
                // 使用多智能体后端 (混合路由模式)
                const agentResponse = await chatWithAgent({
                    message: input.trim(),
                    userId: userId,
                    preferredAgent: 'auto', // Always use auto routing
                    context: {
                        geometry: {
                            camber: geometry.camber,
                            thickness: geometry.thickness,
                            maxCamberPos: geometry.maxCamberPos,
                            maxThicknessPos: geometry.maxThicknessPos,
                            nacaCode: result?.geometry.nacaCode,
                        },
                        environment: {
                            alpha: environment.alpha,
                            re: re,
                            velocity: environment.velocity,
                            rho: environment.rho,
                            mu: environment.mu,
                        },
                        kpi: result?.kpi ? {
                            cl: result.kpi.cl,
                            cd: result.kpi.cd,
                            ld: result.kpi.ld,
                            alphaOpt: result.kpi.alphaOpt,
                            ldMax: result.kpi.ldMax,
                        } : undefined,
                    },
                });

                aiContent = agentResponse.response;
                agentName = agentResponse.agent_display_name;

                // Map internal agent name to module ID if needed (or just use returned display name if it matches)
                // The backend returns mapped names (e.g. "Concept Learning") so we can hopefully use them directly or map them back.
                // Actually agentResponse.agent is the key (e.g. concept_mentor), agentResponse.agent_display_name is "Concept Learning"

                const moduleMap: Record<string, any> = {
                    'concept_mentor': 'Concept Learning',
                    'iteration_engineer': 'Model Iteration',
                    'strategy_analyst': 'Strategy Review'
                };

                const moduleName = moduleMap[agentResponse.agent] || 'Concept Learning';

                // Update the user message to associate it with the handling agent (module)
                updateMessage(userMessage.id, { module: moduleName });

                // Also update current module for UI focus
                setModule(moduleName);

                // Save to backend using the actual agent that responded
                if (userId && userId !== 'guest') {
                    saveConversation(userId, moduleName, userMessage.content, aiContent).catch(err =>
                        console.error('Failed to save conversation:', err)
                    );
                }

                const aiMessage: ChatMessage = {
                    id: uuidv4(),
                    role: 'ai',
                    content: aiContent,
                    module: moduleName,
                    timestamp: new Date().toISOString(),
                };
                addMessage(aiMessage);

            } else {
                // Fallback for non-multi-agent (legacy)
                const simContext = result
                    ? `当前仿真数据：NACA ${result.geometry.nacaCode}, α=${environment.alpha}°, CL=${result.kpi.cl.toFixed(3)}, CD=${result.kpi.cd.toFixed(5)}, L/D=${result.kpi.ld.toFixed(1)}, Re≈${re.toFixed(0)}`
                    : `当前参数：弯度${geometry.camber}%, 厚度${geometry.thickness}%, α=${environment.alpha}°`;

                // ... (Legacy code omitted for brevity as we are using USE_MULTI_AGENT=true)
                // Logic would go here
            }

        } catch (error) {
            console.error('AI API error:', error);
            const errorMessage: ChatMessage = {
                id: uuidv4(),
                role: 'ai',
                content: `抱歉，AI服务暂时不可用。错误：${error instanceof Error ? error.message : 'Unknown error'}`,
                module: currentModule,
                timestamp: new Date().toISOString(),
            };
            addMessage(errorMessage);
        } finally {
            setStreaming(false);
        }
    };

    // 过滤消息
    const filteredMessages = messages.filter(msg => {
        // Role filter - apply to BOTH user and AI messages
        if (roleFilter !== 'ALL' && msg.module !== roleFilter) return false;

        // Search filter
        if (searchTerm.trim()) {
            const term = searchTerm.toLowerCase();
            return msg.content.toLowerCase().includes(term);
        }

        return true;
    });

    // 获取模块样式
    const getModuleStyle = (module?: ChatMessage['module']) => {
        const found = AI_MODULES.find((m) => m.id === module);
        return found?.color || 'bg-white/5 text-slate-300';
    };

    const preprocessContent = (content: string) => {
        if (!content) return '';
        return content
            .replace(/\\\[([\s\S]*?)\\\]/g, '$$$$$1$$$$') // Replace \[ ... \] with $$ ... $$
            .replace(/\\\(([\s\S]*?)\\\)/g, '$$$1$$')     // Replace \( ... \) with $ ... $
            .replace(/<thought>([\s\S]*?)(?:<\/thought>|$)/g, (match, p1) => {
                return '> **🤔 AI 思考过程**\n>\n' + p1.split('\n').map((line: string) => `> ${line}`).join('\n') + '\n\n';
            });
    };

    return (
        <div className={`flex flex-col bg-slate-900 rounded-2xl shadow-lg border border-white/5 overflow-hidden h-full ${className}`}>
            {/* Header */}
            <div className="p-3 border-b border-white/5 flex-shrink-0">
                <div className="flex items-center justify-between mb-2">
                    <h2 className="text-base font-semibold text-slate-200">AI Tutor Chat</h2>
                    <span className="text-[10px] bg-gradient-to-r from-purple-500 to-blue-500 text-white px-2 py-0.5 rounded-full flex items-center gap-1">
                        <Zap className="w-3 h-3" />
                        Multi-Agent
                    </span>
                </div>

                {/* Agent Roles Info (Static with Tooltips) */}
                <div className="grid grid-cols-3 gap-2 mb-2">
                    <TooltipProvider>
                        {AI_MODULES.map((mod) => {
                            const Icon = mod.icon;
                            return (
                                <Tooltip key={mod.id}>
                                    <TooltipTrigger asChild>
                                        <div
                                            className={`flex flex-col items-center justify-center p-2 rounded-lg border border-white/5 ${mod.color} opacity-80 cursor-help transition-opacity hover:opacity-100`}
                                        >
                                            <Icon className={`w-4 h-4 mb-1 ${mod.color.split(' ')[1]}`} />
                                            <span className={`text-[10px] font-medium ${mod.color.split(' ')[1]}`}>{mod.label}</span>
                                        </div>
                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <p className="text-xs">{mod.description}</p>
                                    </TooltipContent>
                                </Tooltip>
                            );
                        })}
                    </TooltipProvider>
                </div>

                {/* Search & Filter Bar */}
                <div className="flex items-center gap-2 mt-2 pt-2 border-t border-white/5">
                    {showSearch ? (
                        <div className="flex-1 flex items-center gap-2 bg-white/5 rounded px-2 py-1">
                            <Search className="w-3 h-3 text-slate-500" />
                            <input
                                className="flex-1 bg-transparent text-xs text-slate-200 border-none focus:ring-0 p-0 placeholder:text-slate-500"
                                placeholder="Search..."
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                autoFocus
                            />
                            <button onClick={() => { setSearchTerm(''); setShowSearch(false); }}><XCircle className="w-3 h-3 text-slate-500 hover:text-slate-300" /></button>
                        </div>
                    ) : (
                        <div className="flex-1"></div>
                    )}

                    <div className="flex items-center gap-1">
                        <Button variant="ghost" size="sm" className="h-6 w-6 p-0 cursor-pointer" onClick={() => setShowSearch(!showSearch)} title="Search">
                            <Search className="w-3.5 h-3.5 text-slate-500" />
                        </Button>
                        <select
                            className="text-xs border border-white/10 rounded px-1 py-0.5 bg-white/5 text-slate-400 focus:outline-none"
                            value={roleFilter}
                            onChange={(e) => setRoleFilter(e.target.value)}
                        >
                            <option value="ALL">All Roles</option>
                            {AI_MODULES.map(m => (
                                <option key={m.id} value={m.id}>{m.label}</option>
                            ))}
                        </select>
                    </div>
                </div>
            </div>

            {/* Messages - fixed height scrollable area */}
            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto p-3 space-y-3"
                style={{}}
            >
                {messages.length === 0 && (
                    <div className="text-center text-slate-500 text-sm py-6">
                        <p>Start a conversation with the AI tutor.</p>
                        <p className="text-xs mt-1 text-slate-600">Ask about aerodynamics or airfoil design.</p>
                    </div>
                )}

                {filteredMessages.length === 0 && (
                    <div className="text-center text-slate-500 text-sm py-6">
                        {messages.length === 0 ? (
                            <>
                                <p>Start a conversation with the AI tutor.</p>
                                <p className="text-xs mt-1 text-slate-600">Ask about aerodynamics or airfoil design.</p>
                            </>
                        ) : (
                            <p>No messages match your search.</p>
                        )}
                    </div>
                )}

                {filteredMessages.map((msg) => (
                    <div
                        key={msg.id}
                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                        <div
                            className={`max-w-[85%] rounded-xl px-3 py-2 text-sm ${msg.role === 'user'
                                ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white rounded-br-sm'
                                : `${getModuleStyle(msg.module)} rounded-bl-sm`
                                }`}
                        >
                            {msg.module && msg.role === 'ai' && (
                                <div className="text-[9px] font-medium opacity-60 mb-0.5">
                                    {msg.module}
                                </div>
                            )}
                            <div className="text-sm leading-relaxed overflow-hidden">
                                <ReactMarkdown
                                    remarkPlugins={[remarkGfm, remarkMath]}
                                    rehypePlugins={[rehypeKatex]}
                                    components={{
                                        p: ({ node, ...props }: any) => <p className="mb-2 last:mb-0" {...props} />,
                                        ul: ({ node, ...props }: any) => <ul className="list-disc pl-4 mb-2 space-y-1" {...props} />,
                                        ol: ({ node, ...props }: any) => <ol className="list-decimal pl-4 mb-2 space-y-1" {...props} />,
                                        li: ({ node, ...props }: any) => <li className="pl-1" {...props} />,
                                        a: ({ node, ...props }: any) => <a className="underline opacity-80 hover:opacity-100" target="_blank" {...props} />,
                                        code({ node, className, children, ...props }: any) {
                                            const match = /language-(\w+)/.exec(className || '');
                                            const isInline = !match && !String(children).includes('\n');

                                            // Check if it's JSON from the Iteration Engineer
                                            const codeString = String(children);
                                            let isParameterJson = false;
                                            let parsedParams: Partial<GeometryParams> | null = null;

                                            if (!isInline && match?.[1] === 'json') {
                                                try {
                                                    const parsed = JSON.parse(codeString);
                                                    if ('camber' in parsed || 'thickness' in parsed || 'maxCamberPos' in parsed) {
                                                        isParameterJson = true;
                                                        parsedParams = parsed;
                                                    }
                                                } catch (e) {
                                                    // Not a valid JSON or doesn't match our schema
                                                }
                                            }

                                            return isInline ? (
                                                <code className="bg-white/10 px-1 py-0.5 rounded text-xs font-mono text-slate-200" {...props}>
                                                    {children}
                                                </code>
                                            ) : (
                                                <div className="my-2 rounded-md overflow-hidden bg-black/30 text-slate-200 border border-white/10">
                                                    <div className="px-3 py-2 bg-black/20 text-xs text-slate-400 font-mono flex items-center justify-between border-b border-white/5">
                                                        <span className="flex items-center gap-2">
                                                            <div className="w-2 h-2 rounded-full bg-slate-600"></div>
                                                            {isParameterJson ? 'Suggested Parameters' : (match?.[1] || 'code')}
                                                        </span>
                                                        {isParameterJson && parsedParams && (
                                                            <Button
                                                                size="sm"
                                                                variant="default"
                                                                className="h-6 text-[10px] px-2 bg-amber-600 hover:bg-amber-500 text-white"
                                                                onClick={() => {
                                                                    if (parsedParams) setGeometry(parsedParams);
                                                                }}
                                                            >
                                                                <Target className="w-3 h-3 mr-1" />
                                                                Apply to Airfoil
                                                            </Button>
                                                        )}
                                                    </div>
                                                    <div className="p-3 overflow-x-auto">
                                                        <code className={`text-xs font-mono leading-relaxed ${className || ''}`} {...props}>
                                                            {children}
                                                        </code>
                                                    </div>
                                                </div>
                                            );
                                        },
                                        table: ({ node, ...props }: any) => <div className="overflow-x-auto my-2 rounded border border-white/10"><table className="min-w-full divide-y divide-white/10 text-xs" {...props} /></div>,
                                        thead: ({ node, ...props }: any) => <thead className="bg-white/5" {...props} />,
                                        th: ({ node, ...props }: any) => <th className="px-3 py-2 text-left font-medium opacity-80" {...props} />,
                                        tbody: ({ node, ...props }: any) => <tbody className="divide-y divide-white/5" {...props} />,
                                        td: ({ node, ...props }: any) => <td className="px-3 py-2 whitespace-nowrap border-r border-white/5 last:border-r-0" {...props} />,
                                        blockquote: ({ node, ...props }: any) => <blockquote className="border-l-2 border-white/20 pl-3 italic my-2 opacity-80" {...props} />,
                                    }}
                                >
                                    {preprocessContent(msg.content)}
                                </ReactMarkdown>
                            </div>
                        </div>
                    </div>
                ))}

                {isStreaming && (
                    <div className="flex justify-start">
                        <div className="bg-white/5 rounded-xl rounded-bl-sm px-3 py-2 flex items-center gap-2">
                            <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
                            {USE_MULTI_AGENT && (
                                <span className="text-xs text-slate-500">智能体思考中...</span>
                            )}
                        </div>
                    </div>
                )}
            </div>

            {/* Input Area */}
            <div className="p-3 border-t border-white/5 flex-shrink-0">
                <div className="flex gap-2 items-center">
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={insertSimData}
                        className="text-xs px-2 h-8 cursor-pointer hover:bg-white/5"
                        title="Insert simulation data"
                    >
                        📊
                    </Button>
                    <Textarea
                        ref={inputRef as any}
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handleSend();
                            }
                        }}
                        placeholder="Ask about airfoil design..."
                        className="flex-1 min-h-[40px] max-h-[120px] resize-none text-sm py-2 px-3 bg-white/5 border-white/10 text-slate-200 placeholder:text-slate-500 focus-visible:ring-0 focus-visible:border-blue-500/50"
                        disabled={isStreaming}
                        rows={1}
                        style={{ height: 'auto', minHeight: '40px' }}
                        onInput={(e) => {
                            const target = e.target as HTMLTextAreaElement;
                            target.style.height = 'auto';
                            target.style.height = `${Math.min(target.scrollHeight, 120)}px`;
                        }}
                    />
                    <Button
                        onClick={handleSend}
                        disabled={!input.trim() || isStreaming}
                        size="sm"
                        className="rounded-lg px-3 h-8 bg-blue-600 hover:bg-blue-500 cursor-pointer"
                    >
                        {isStreaming ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                            <Send className="w-4 h-4" />
                        )}
                    </Button>
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setInput('')}
                        className="text-xs px-2 h-8 text-slate-500 hover:text-slate-300 hover:bg-white/5 cursor-pointer"
                        title="Clear input"
                        disabled={!input}
                    >
                        <XCircle className="w-3.5 h-3.5" />
                    </Button>
                </div>
            </div>
        </div >
    );
}

export default ChatWindow;
