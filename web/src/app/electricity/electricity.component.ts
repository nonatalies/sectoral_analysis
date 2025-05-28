import { Component } from '@angular/core';
import {AppService} from '../app.service';
import {MatSlideToggleChange} from '@angular/material/slide-toggle';
import { LegendPosition } from '@swimlane/ngx-charts';

@Component({
  selector: 'app-electricity',
  standalone: false,
  templateUrl: './electricity.component.html',
  styleUrl: './electricity.component.scss'
})
export class ElectricityComponent {
  private dataOld1: any[] = [];
  private dataOld2: any[] = [];
  private dataOld3: any[] = [];
  private dataNew1: any[] = [];
  private dataNew2: any[] = [];
  private dataNew3: any[] = [];
  public chartData1: {name: string, value: number, type: boolean}[] = [];
  public chartData2: {name: string, value: number, type: boolean, originalValue?: number}[] = [];
  public chartData3: {name: string, series: {name: string, x: number, y: number, r: number}, type: boolean}[] = [];
  public chartData4: {name: string, value: number}[] = [{name: 'Существующий', value: 14118.8}, {name: 'Новый', value: 7259.7}];
  public chartData5: {name: string, value: number, originalValue?: number}[] = [{name: 'Существующий', value: 8902.6}, {name: 'Новый', value: 1658.2}];

  public newTurnOn: boolean = false;
  public bubbleTurnOn: boolean = true;

  constructor(private appService: AppService) {
    this.appService.districtData$.subscribe({
      next: (data) => {
        const data1 = data
          .map(it => ({
            name: it['Регион'],
            value: it['Потребление электроэнергии (млн кВт·ч)'],
            type: it['Тип'] === 'Новый'
          }));

        const data2 = data
          .map(it => ({
            name: it['Регион'],
            value: it['Производство электроэнергии на душу (кВт·ч/чел)'],
            type: it['Тип'] === 'Новый',
          }));

        const data3 = data
          .map(it => ({
            name: it['Регион'],
            series: [
              {
                name: it['Регион'],
                x: it['Потребление электроэнергии (млн кВт·ч)'],
                y: it['Производство электроэнергии на душу (кВт·ч/чел)'],
                r: 1000000
              },
            ],
            type: it['Тип'] === 'Новый',
          }));

        this.dataOld1 = data1.filter((it) => !it.type);
        this.dataOld2 = data2.filter((it) => !it.type);
        this.dataOld3 = data3.filter((it) => !it.type);
        this.dataNew1 = data1.filter((it) => it.type);
        this.dataNew2 = data2.filter((it) => it.type);
        this.dataNew3 = data3.filter((it) => it.type);

        this.chartData1 = this.dataOld1;
        this.chartData2 = this.dataOld2;
        this.chartData3 = this.dataOld3;
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
    this.mouseX = event.clientX -80;
    this.mouseY = event.clientY -80;
  }

  onToggleChange(event: MatSlideToggleChange): void {
    if (event.checked) {
      this.chartData1 = this.dataNew1;
      this.chartData2 = this.dataNew2;
      this.chartData3 = this.dataNew3;
    } else {
      this.chartData1 = this.dataOld1;
      this.chartData2 = this.dataOld2;
      this.chartData3 = this.dataOld3;
    }
  }

}
