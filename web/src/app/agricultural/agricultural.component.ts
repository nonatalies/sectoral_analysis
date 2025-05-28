import { Component } from '@angular/core';
import { AppService } from '../app.service';
import {LegendPosition} from '@swimlane/ngx-charts';
import {MatSlideToggleChange} from '@angular/material/slide-toggle';

@Component({
  selector: 'app-agricultural',
  standalone: false,
  templateUrl: './agricultural.component.html',
  styleUrl: './agricultural.component.scss'
})
export class AgriculturalComponent {
  public dataOld1: any[] = [];
  public dataOld2: any[] = [];
  private dataNew1: any[] = [];
  private dataNew2: any[] = [];
  public chartData1: {name: string, value: number, type: boolean}[] = [];
  public chartData2: {name: string, value: number, type: boolean, originalValue?: number}[] = [];
  public chartData3: {name: string, value: number}[] = [{name: 'Существующий', value: 942.5}, {name: 'Новый', value: 908.1}];
  public chartData4: {name: string, value: number, originalValue?: number}[] = [{name: 'Существующий', value: 1.9, originalValue: 101.9}, {name: 'Новый', value: -2.4, originalValue: 97.6}];

  public newTurnOn: boolean = false;

  constructor(private appService: AppService) {
    this.appService.districtData$.subscribe({
      next: (data) => {
        const data1 = data
          .map(it => ({
            name: it['Регион'],
            value: it['Посевные площади (тыс. га)'],
            type: it['Тип'] === 'Новый'
          }));

        const data2 = data
          .map(it => ({
            name: it['Регион'],
            value: it['Индекс сельского хозяйства'] - 100,
            type: it['Тип'] === 'Новый',
            originalValue: it['Индекс сельского хозяйства']
          }));

        this.dataOld1 = data1.filter((it) => !it.type);
        this.dataOld2 = data2.filter((it) => !it.type);
        this.dataNew1 = data1.filter((it) => it.type);
        this.dataNew2 = data2.filter((it) => it.type);

        this.chartData1 = this.dataOld1;
        this.chartData2 = this.dataOld2;
      },
      error: (error) => {
        console.error('Error in component subscription:', error);
      }
    });
  }

  view: [number, number] = [700, 270];
  smallView: [number, number] = [500, 200];

  // options
  showXAxis = false;
  showYAxis = true;
  gradient = false;
  showLegend = true;
  protected readonly LegendPosition = LegendPosition;


  tooltipData: any = null;

  onActivate(event: any, data: any[]): void {
    this.tooltipData = data.find((it) => it.name === event.value.name);
  }

  onDeactivate(): void {
    this.tooltipData = null;
  }
  mouseX = 0;
  mouseY = 0;

  onMouseMove(event: MouseEvent): void {
    this.mouseX = event.clientX -60;
    this.mouseY = event.clientY -60;
  }

  onToggleChange(event: MatSlideToggleChange): void {
    if (event.checked) {
      this.chartData1 = this.dataNew1;
      this.chartData2 = this.dataNew2;
    } else {
      this.chartData1 = this.dataOld1;
      this.chartData2 = this.dataOld2;
    }
  }
}
