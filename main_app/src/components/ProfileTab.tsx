import { useState } from 'react';
import { Card } from './ui/card';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Button } from './ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { ScrollArea } from './ui/scroll-area';
import { User, Mail, Phone, MapPin, Calendar, Edit2, Save, Moon, Sun } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

export function ProfileTab() {
  const [isEditing, setIsEditing] = useState(false);
  const { theme, toggleTheme } = useTheme();
  const [profile, setProfile] = useState({
    firstName: 'Алия',
    lastName: 'Нурбекова',
    age: '28',
    gender: 'female',
    email: 'aliya.nurbekova@example.com',
    phone: '+7 707 123 4567',
    city: 'Алматы',
    dateOfBirth: '1997-03-15'
  });

  const handleSave = () => {
    setIsEditing(false);
    // Save profile changes
  };

  return (
    <ScrollArea className="h-full bg-gray-50 dark:bg-gray-900">
      <div className="p-4 pb-6">
        {/* Profile Header */}
        <div className="bg-gradient-to-r from-[#2D9A86] to-[#EEFE6D] rounded-t-lg p-6 text-center">
          <div className="w-24 h-24 bg-white rounded-full mx-auto mb-4 flex items-center justify-center">
            <User className="w-12 h-12 text-[#2D9A86]" />
          </div>
          <h1 className="text-2xl">{profile.firstName} {profile.lastName}</h1>
          <p className="text-sm opacity-90 mt-1">{profile.email}</p>
        </div>

        <Card className="rounded-t-none p-6 dark:bg-gray-800 dark:border-gray-700">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-lg dark:text-white">Личная информация</h2>
            {!isEditing ? (
              <Button
                onClick={() => setIsEditing(true)}
                variant="outline"
                size="sm"
                className="gap-2 dark:border-gray-600 dark:text-white"
              >
                <Edit2 className="w-4 h-4" />
                Редактировать
              </Button>
            ) : (
              <Button
                onClick={handleSave}
                size="sm"
                className="gap-2 bg-[#2D9A86] hover:bg-[#268976]"
              >
                <Save className="w-4 h-4" />
                Сохранить
              </Button>
            )}
          </div>

          <div className="space-y-4">
            {/* First Name */}
            <div>
              <Label htmlFor="firstName" className="dark:text-gray-300">Имя</Label>
              <div className="flex items-center gap-2 mt-1">
                <User className="w-5 h-5 text-gray-400" />
                <Input
                  id="firstName"
                  value={profile.firstName}
                  onChange={(e) => setProfile({ ...profile, firstName: e.target.value })}
                  disabled={!isEditing}
                  className="flex-1 dark:bg-gray-700 dark:text-white dark:border-gray-600"
                />
              </div>
            </div>

            {/* Last Name */}
            <div>
              <Label htmlFor="lastName" className="dark:text-gray-300">Фамилия</Label>
              <div className="flex items-center gap-2 mt-1">
                <User className="w-5 h-5 text-gray-400" />
                <Input
                  id="lastName"
                  value={profile.lastName}
                  onChange={(e) => setProfile({ ...profile, lastName: e.target.value })}
                  disabled={!isEditing}
                  className="flex-1 dark:bg-gray-700 dark:text-white dark:border-gray-600"
                />
              </div>
            </div>

            {/* Date of Birth */}
            <div>
              <Label htmlFor="dob" className="dark:text-gray-300">Дата рождения</Label>
              <div className="flex items-center gap-2 mt-1">
                <Calendar className="w-5 h-5 text-gray-400" />
                <Input
                  id="dob"
                  type="date"
                  value={profile.dateOfBirth}
                  onChange={(e) => setProfile({ ...profile, dateOfBirth: e.target.value })}
                  disabled={!isEditing}
                  className="flex-1 dark:bg-gray-700 dark:text-white dark:border-gray-600"
                />
              </div>
            </div>

            {/* Age */}
            <div>
              <Label htmlFor="age" className="dark:text-gray-300">Возраст</Label>
              <div className="flex items-center gap-2 mt-1">
                <Calendar className="w-5 h-5 text-gray-400" />
                <Input
                  id="age"
                  type="number"
                  value={profile.age}
                  onChange={(e) => setProfile({ ...profile, age: e.target.value })}
                  disabled={!isEditing}
                  className="flex-1 dark:bg-gray-700 dark:text-white dark:border-gray-600"
                />
              </div>
            </div>

            {/* Gender */}
            <div>
              <Label htmlFor="gender" className="dark:text-gray-300">Пол</Label>
              <Select
                value={profile.gender}
                onValueChange={(value) => setProfile({ ...profile, gender: value })}
                disabled={!isEditing}
              >
                <SelectTrigger className="mt-1 dark:bg-gray-700 dark:text-white dark:border-gray-600">
                  <SelectValue placeholder="Выберите пол" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="female">Женский</SelectItem>
                  <SelectItem value="male">Мужской</SelectItem>
                  <SelectItem value="other">Другое</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Email */}
            <div>
              <Label htmlFor="email" className="dark:text-gray-300">Email</Label>
              <div className="flex items-center gap-2 mt-1">
                <Mail className="w-5 h-5 text-gray-400" />
                <Input
                  id="email"
                  type="email"
                  value={profile.email}
                  onChange={(e) => setProfile({ ...profile, email: e.target.value })}
                  disabled={!isEditing}
                  className="flex-1 dark:bg-gray-700 dark:text-white dark:border-gray-600"
                />
              </div>
            </div>

            {/* Phone */}
            <div>
              <Label htmlFor="phone" className="dark:text-gray-300">Телефон</Label>
              <div className="flex items-center gap-2 mt-1">
                <Phone className="w-5 h-5 text-gray-400" />
                <Input
                  id="phone"
                  type="tel"
                  value={profile.phone}
                  onChange={(e) => setProfile({ ...profile, phone: e.target.value })}
                  disabled={!isEditing}
                  className="flex-1 dark:bg-gray-700 dark:text-white dark:border-gray-600"
                />
              </div>
            </div>

            {/* City */}
            <div>
              <Label htmlFor="city" className="dark:text-gray-300">Город</Label>
              <div className="flex items-center gap-2 mt-1">
                <MapPin className="w-5 h-5 text-gray-400" />
                <Input
                  id="city"
                  value={profile.city}
                  onChange={(e) => setProfile({ ...profile, city: e.target.value })}
                  disabled={!isEditing}
                  className="flex-1 dark:bg-gray-700 dark:text-white dark:border-gray-600"
                />
              </div>
            </div>
          </div>
        </Card>

        {/* Theme Toggle */}
        <Card className="p-6 mt-4 dark:bg-gray-800 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium dark:text-white">Тема</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                {theme === 'light' ? 'Светлая' : 'Тёмная'} тема
              </p>
            </div>
            <Button
              onClick={toggleTheme}
              variant="outline"
              size="icon"
              className="dark:border-gray-600"
            >
              {theme === 'light' ? (
                <Moon className="w-5 h-5" />
              ) : (
                <Sun className="w-5 h-5" />
              )}
            </Button>
          </div>
        </Card>

        {/* Statistics Card */}
        <Card className="p-6 mt-4 dark:bg-gray-800 dark:border-gray-700">
          <h2 className="text-lg mb-4 dark:text-white">Статистика</h2>
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-2xl text-[#2D9A86]">342</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">Всего транзакций</div>
            </div>
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-2xl text-[#2D9A86]">16</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">Категорий используется</div>
            </div>
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-2xl text-[#2D9A86]">89%</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">Автокатегоризация</div>
            </div>
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div className="text-2xl text-[#2D9A86]">6 мес</div>
              <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">С нами</div>
            </div>
          </div>
        </Card>
      </div>
    </ScrollArea>
  );
}
