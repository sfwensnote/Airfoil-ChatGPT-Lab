'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useUserStore } from '@/stores';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Trash2, Plus, Shield, ShieldCheck, LogOut, ArrowLeft } from 'lucide-react';
import { toast } from 'sonner';
import { Header } from '@/components/layout/Header';

interface UserData {
    id: number;
    username: string;
    password?: string;
    role: string;
}

export default function AdminPage() {
    const { currentUser, isAuthenticated, logout } = useUserStore();
    const router = useRouter();
    const [users, setUsers] = useState<UserData[]>([]);
    const [loading, setLoading] = useState(true);
    const [newUsername, setNewUsername] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [isDialogOpen, setIsDialogOpen] = useState(false);

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    useEffect(() => {
        if (!isAuthenticated || currentUser?.role !== 'admin') {
            router.push('/login');
            return;
        }
        fetchUsers();
    }, [isAuthenticated, currentUser, router]);

    const fetchUsers = async () => {
        try {
            const res = await fetch(`${apiUrl}/admin/users`);
            const data = await res.json();
            setUsers(data);
        } catch (err) {
            console.error(err);
            toast.error('Failed to fetch users');
        } finally {
            setLoading(false);
        }
    };

    const handleAddUser = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            const res = await fetch(`${apiUrl}/admin/users`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username: newUsername, password: newPassword }),
            });
            const data = await res.json();
            if (data.status === 'success') {
                toast.success('User added successfully');
                setNewUsername('');
                setNewPassword('');
                setIsDialogOpen(false);
                fetchUsers();
            } else {
                toast.error(data.message || 'Failed to add user');
            }
        } catch (err) {
            toast.error('Error connecting to server');
        }
    };

    const handleDeleteUser = async (id: number) => {
        if (!confirm('Are you sure you want to delete this user?')) return;
        try {
            const res = await fetch(`${apiUrl}/admin/users/${id}`, { method: 'DELETE' });
            const data = await res.json();
            if (data.status === 'success') {
                toast.success('User deleted');
                fetchUsers();
            } else {
                toast.error(data.message);
            }
        } catch (err) {
            toast.error('Delete failed');
        }
    };

    const handleLogout = () => {
        logout();
        router.push('/login');
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-[hsl(220,20%,10%)] flex items-center justify-center">
                <p className="text-slate-500">Loading...</p>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
            <Header />
            <div className="max-w-6xl mx-auto px-6 py-8 space-y-8">
                {/* Header */}
                <div className="flex justify-between items-center">
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight text-white">Admin Dashboard</h1>
                        <p className="text-slate-400 mt-2">Manage laboratory access and user accounts.</p>
                    </div>
                    <div className="flex gap-2">
                        <Button variant="outline" onClick={() => router.push('/')}
                            className="border-[hsl(220,15%,25%)] text-slate-200 hover:bg-[hsl(220,15%,20%)] cursor-pointer">
                            <ArrowLeft className="mr-2 h-4 w-4" /> Go to Lab
                        </Button>
                        <Button variant="destructive" onClick={handleLogout} className="cursor-pointer">
                            <LogOut className="mr-2 h-4 w-4" /> Sign Out
                        </Button>
                    </div>
                </div>

                {/* Users Management */}
                <div className="card-panel p-6">
                    <div className="flex items-center justify-between mb-6">
                        <div>
                            <h2 className="text-lg font-semibold text-white">Registered Users</h2>
                            <p className="text-sm text-slate-400">Total Users: {users.length}</p>
                        </div>
                        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                            <DialogTrigger asChild>
                                <Button className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 border-0 cursor-pointer">
                                    <Plus className="mr-2 h-4 w-4" /> Add User
                                </Button>
                            </DialogTrigger>
                            <DialogContent className="bg-[hsl(220,20%,14%)] border-[hsl(220,15%,22%)]">
                                <DialogHeader>
                                    <DialogTitle className="text-white">Add New User</DialogTitle>
                                </DialogHeader>
                                <form onSubmit={handleAddUser} className="space-y-4 py-4">
                                    <div className="space-y-2">
                                        <Label htmlFor="new-username" className="text-slate-300">Username</Label>
                                        <Input
                                            id="new-username"
                                            value={newUsername}
                                            onChange={(e) => setNewUsername(e.target.value)}
                                            placeholder="Enter username"
                                            className="bg-[hsl(220,15%,18%)] border-[hsl(220,15%,25%)] text-white"
                                            required
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <Label htmlFor="new-password" className="text-slate-300">Password</Label>
                                        <Input
                                            id="new-password"
                                            value={newPassword}
                                            onChange={(e) => setNewPassword(e.target.value)}
                                            placeholder="Enter password"
                                            className="bg-[hsl(220,15%,18%)] border-[hsl(220,15%,25%)] text-white"
                                            required
                                        />
                                        <p className="text-xs text-slate-500">Passwords are stored as plain text per requirements.</p>
                                    </div>
                                    <Button type="submit" className="w-full bg-gradient-to-r from-blue-600 to-purple-600 border-0 cursor-pointer">
                                        Create Account
                                    </Button>
                                </form>
                            </DialogContent>
                        </Dialog>
                    </div>

                    <div className="overflow-x-auto rounded-lg border border-[hsl(220,15%,22%)]">
                        <table className="min-w-full text-sm">
                            <thead className="bg-[hsl(220,15%,18%)]">
                                <tr>
                                    <th className="px-4 py-3 text-left text-slate-300 font-medium">ID</th>
                                    <th className="px-4 py-3 text-left text-slate-300 font-medium">Role</th>
                                    <th className="px-4 py-3 text-left text-slate-300 font-medium">Username</th>
                                    <th className="px-4 py-3 text-left text-slate-300 font-medium">Password</th>
                                    <th className="px-4 py-3 text-right text-slate-300 font-medium">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-[hsl(220,15%,20%)]">
                                {users.map((user) => (
                                    <tr key={user.id} className="hover:bg-[hsl(220,15%,18%)]">
                                        <td className="px-4 py-3 text-slate-300 font-mono">{user.id}</td>
                                        <td className="px-4 py-3">
                                            <Badge variant={user.role === 'admin' ? 'default' : 'secondary'}
                                                className={user.role === 'admin'
                                                    ? 'bg-blue-500/20 text-blue-300 border-blue-500/30'
                                                    : 'bg-[hsl(220,15%,20%)] text-slate-300 border-[hsl(220,15%,25%)]'
                                                }>
                                                {user.role === 'admin' ? <ShieldCheck className="h-3 w-3 mr-1" /> : <Shield className="h-3 w-3 mr-1" />}
                                                {user.role}
                                            </Badge>
                                        </td>
                                        <td className="px-4 py-3 font-medium text-white">{user.username}</td>
                                        <td className="px-4 py-3 font-mono text-slate-400">{user.password}</td>
                                        <td className="px-4 py-3 text-right">
                                            {user.username !== 'admin' && (
                                                <Button
                                                    variant="ghost"
                                                    size="icon"
                                                    className="text-red-400 hover:text-red-300 hover:bg-red-500/10 cursor-pointer"
                                                    onClick={() => handleDeleteUser(user.id)}
                                                >
                                                    <Trash2 className="h-4 w-4" />
                                                </Button>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    );
}
