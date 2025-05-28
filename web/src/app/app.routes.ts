import { Routes } from '@angular/router';
import {IndustryComponent} from './industry/industry.component';
import {ElectricityComponent} from './electricity/electricity.component';
import {AgriculturalComponent} from './agricultural/agricultural.component';
import {TableComponent} from './table/table.component';
import {StatComponent} from './stat/stat.component';

export const routes: Routes = [
  { path: '', redirectTo: 'agricultural', pathMatch: 'full' },
  { path: 'agricultural', component: AgriculturalComponent },
  { path: 'electricity', component: ElectricityComponent },
  { path: 'industry', component: IndustryComponent },
  { path: 'stat', component: StatComponent },
  { path: 'table', component: TableComponent },
  { path: '**', redirectTo: 'agricultural' }
];
