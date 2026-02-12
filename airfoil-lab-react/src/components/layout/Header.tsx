'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    Plane,
    History,
    HelpCircle,
    Shield,
    Menu,
    X,
    LogOut
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import { useState } from 'react';
import { useUserStore } from '@/stores';

const navItems = [
    { href: '/', label: 'Geometry', icon: Plane },
    { href: '/history', label: 'History', icon: History },
    { href: '/help', label: 'Help', icon: HelpCircle },
    { href: '/admin', label: 'Admin', icon: Shield },
];

interface HeaderProps {
    className?: string;
}

export function Header({ className = '' }: HeaderProps) {
    const pathname = usePathname();
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    const { currentUser, isAuthenticated, logout } = useUserStore();

    return (
        <header className={`bg-[hsl(220,20%,12%)] border-b border-[hsl(220,15%,20%)] sticky top-0 z-40 ${className}`}>
            <div className="max-w-7xl mx-auto px-4 sm:px-6">
                <div className="flex justify-between items-center h-12">
                    {/* Logo */}
                    <div className="flex items-center gap-2.5">
                        <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center shadow-lg">
                            <Plane className="w-4 h-4 text-white" />
                        </div>
                        <div>
                            <h1 className="text-sm font-bold text-slate-100 leading-tight">Airfoil Lab</h1>
                            <p className="text-[10px] text-slate-400 leading-tight">AI-Enhanced Design</p>
                        </div>
                    </div>

                    {/* Desktop Navigation - Icon only with tooltips */}
                    <TooltipProvider delayDuration={200}>
                        <nav className="hidden md:flex items-center gap-1">
                            {navItems.map((item) => {
                                const Icon = item.icon;
                                const isActive = pathname === item.href;
                                const isAdminRoute = item.href === '/admin';

                                if (isAdminRoute && currentUser?.role !== 'admin') return null;

                                return (
                                    <Tooltip key={item.href}>
                                        <TooltipTrigger asChild>
                                            <Link href={item.href}>
                                                <Button
                                                    variant={isActive ? 'default' : 'ghost'}
                                                    size="sm"
                                                    className={`w-8 h-8 p-0 cursor-pointer ${isActive
                                                        ? 'bg-blue-600/20 text-blue-400 hover:bg-blue-600/30'
                                                        : 'text-slate-300 hover:text-white hover:bg-[hsl(220,15%,20%)]'
                                                        }`}
                                                >
                                                    <Icon className="w-4 h-4" />
                                                </Button>
                                            </Link>
                                        </TooltipTrigger>
                                        <TooltipContent side="bottom" className="text-xs">
                                            {item.label}
                                        </TooltipContent>
                                    </Tooltip>
                                );
                            })}
                        </nav>
                    </TooltipProvider>

                    {/* User area */}
                    <div className="hidden md:flex items-center gap-2">
                        {isAuthenticated ? (
                            <div className="flex items-center gap-2">
                                <div className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center text-xs font-bold text-white uppercase">
                                    {currentUser?.username?.charAt(0) || 'U'}
                                </div>
                                <span className="text-xs font-medium text-slate-300">
                                    {currentUser?.username}
                                </span>
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => logout()}
                                    className="w-7 h-7 p-0 text-slate-400 hover:text-slate-200 hover:bg-[hsl(220,15%,20%)] cursor-pointer"
                                >
                                    <LogOut className="w-3.5 h-3.5" />
                                </Button>
                            </div>
                        ) : (
                            <Link href="/login">
                                <Button size="sm" className="h-7 text-xs px-3 cursor-pointer">Sign In</Button>
                            </Link>
                        )}
                    </div>

                    {/* Mobile menu button */}
                    <Button
                        variant="ghost"
                        size="sm"
                        className="md:hidden w-8 h-8 p-0 text-slate-400 hover:text-white cursor-pointer"
                        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                    >
                        {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                    </Button>
                </div>
            </div>

            {/* Mobile Navigation */}
            {mobileMenuOpen && (
                <div className="md:hidden border-t border-[hsl(220,15%,20%)] bg-[hsl(220,20%,12%)]">
                    <div className="px-4 py-3 space-y-1">
                        {navItems.map((item) => {
                            const Icon = item.icon;
                            const isActive = pathname === item.href;
                            const isAdminRoute = item.href === '/admin';
                            if (isAdminRoute && currentUser?.role !== 'admin') return null;

                            return (
                                <Link key={item.href} href={item.href}>
                                    <Button
                                        variant={isActive ? 'default' : 'ghost'}
                                        className={`w-full justify-start gap-2 cursor-pointer ${isActive
                                            ? 'bg-blue-600/20 text-blue-400'
                                            : 'text-slate-300 hover:text-white hover:bg-[hsl(220,15%,20%)]'
                                            }`}
                                        onClick={() => setMobileMenuOpen(false)}
                                    >
                                        <Icon className="w-4 h-4" />
                                        {item.label}
                                    </Button>
                                </Link>
                            );
                        })}
                        <div className="pt-2 border-t border-[hsl(220,15%,20%)]">
                            {isAuthenticated ? (
                                <div className="space-y-2">
                                    <div className="px-3 py-2 text-xs text-slate-500">
                                        Signed in as <span className="text-slate-300">{currentUser?.username}</span>
                                    </div>
                                    <Button
                                        variant="ghost"
                                        className="w-full justify-start gap-2 text-red-400 hover:text-red-300 hover:bg-red-500/10 cursor-pointer"
                                        onClick={() => {
                                            logout();
                                            setMobileMenuOpen(false);
                                        }}
                                    >
                                        <LogOut className="w-4 h-4" />
                                        Sign Out
                                    </Button>
                                </div>
                            ) : (
                                <Link href="/login" onClick={() => setMobileMenuOpen(false)}>
                                    <Button className="w-full cursor-pointer">Sign In</Button>
                                </Link>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </header>
    );
}

export default Header;
