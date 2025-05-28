import { Component } from '@angular/core';
import {AppService} from '../app.service';
import {LegendPosition} from '@swimlane/ngx-charts';

@Component({
  selector: 'app-stat',
  standalone: false,
  templateUrl: './stat.component.html',
  styleUrl: './stat.component.scss'
})
export class StatComponent {

  public lineData1: any[] =[];
  public lineData2: any[] =[];
  public lineData3: any[] =[];
  public lineData4: any[] =[];
  public lineData5: any[] =[];
  public lineData6: any[] =[];

  public bubbleData1: any[] =[];
  public bubbleData2: any[] =[];
  public bubbleData3: any[] =[];
  public bubbleData4: any[] =[];
  public bubbleData5: any[] =[];
  public bubbleData6: any[] =[];

  view: [number, number] = [600, 270];
  smallView: [number, number] = [600, 267];

  constructor(private appService: AppService) {
    this.appService.trendData$.subscribe({
      next: (data) => {
        this.bubbleData1 = data
          .map(it => ({
            name: it['Регион'],
            series: [
              {
                name: it['Регион'],
                x: it['Тренд 1'],
                y: it['Рост 1'],
                r: 1000000
              },
            ],
          }));
        this.bubbleData2 = data
          .map(it => ({
            name: it['Регион'],
            series: [
              {
                name: it['Регион'],
                x: it['Тренд 2'],
                y: it['Рост 2'],
                r: 1000000
              },
            ],
          }));
        this.bubbleData3 = data
          .map(it => ({
            name: it['Регион'],
            series: [
              {
                name: it['Регион'],
                x: it['Тренд 3'],
                y: it['Рост 3'],
                r: 1000000
              },
            ],
          }));
        this.bubbleData4 = data
          .map(it => ({
            name: it['Регион'],
            series: [
              {
                name: it['Регион'],
                x: it['Тренд 4'],
                y: it['Рост 4'],
                r: 1000000
              },
            ],
          }));
        this.bubbleData5 = data
          .map(it => ({
            name: it['Регион'],
            series: [
              {
                name: it['Регион'],
                x: it['Тренд 5'],
                y: it['Рост 5'],
                r: 1000000
              },
            ],
          }));
        this.bubbleData6 = data
          .map(it => ({
            name: it['Регион'],
            series: [
              {
                name: it['Регион'],
                x: it['Тренд 6'],
                y: it['Рост 6'],
                r: 1000000
              },
            ],
          }));

      },
      error: (error) => {
        console.error('Error in component subscription:', error);
      }
    });

    this.appService.statData$.subscribe({
      next: (data) => {

        const data1 = data
          .map(it => ({
            name: it['Регион'],
            series: [
              {
                name: '2020',
                value: it['Посевные площади (тыс. га) 2020'],
              },
              {
                name: '2021',
                value: it['Посевные площади (тыс. га) 2021'],
              },
              {
                name: '2022',
                value: it['Посевные площади (тыс. га) 2022'],
              },
            ],
          }));
        const data2 = data
          .map(it => ({
            name: it['Регион'],
            series: [
              {
                name: '2020',
                value: it['Индекс сельского хозяйства 2020'],
              },
              {
                name: '2021',
                value: it['Индекс сельского хозяйства 2021'],
              },
              {
                name: '2022',
                value: it['Индекс сельского хозяйства 2022'],
              },
            ],
          }));
        const data3 = data
          .map(it => ({
            name: it['Регион'],
            series: [
              {
                name: '2020',
                value: it['Добыча полезных ископаемых (млн руб) 2020'],
              },
              {
                name: '2021',
                value: it['Добыча полезных ископаемых (млн руб) 2021'],
              },
              {
                name: '2022',
                value: it['Добыча полезных ископаемых (млн руб) 2022'],
              },
            ],
          }));
        const data4 = data
          .map(it => ({
            name: it['Регион'],
            series: [
              {
                name: '2020',
                value: it['Индекс пром. производства 2020'],
              },
              {
                name: '2021',
                value: it['Индекс пром. производства 2021'],
              },
              {
                name: '2022',
                value: it['Индекс пром. производства 2022'],
              },
            ],
          }));
        const data5 = data
          .map(it => ({
            name: it['Регион'],
            series: [
              {
                name: '2020',
                value: it['Потребление электроэнергии (млн кВт·ч) 2020'],
              },
              {
                name: '2021',
                value: it['Потребление электроэнергии (млн кВт·ч) 2021'],
              },
              {
                name: '2022',
                value: it['Потребление электроэнергии (млн кВт·ч) 2022'],
              },
            ],
          }));

        const data6 = data
          .map(it => ({
            name: it['Регион'],
            series: [
              {
                name: '2020',
                value: it['Производство электроэнергии на душу (кВт·ч/чел) 2020'],
              },
              {
                name: '2021',
                value: it['Производство электроэнергии на душу (кВт·ч/чел) 2021'],
              },
              {
                name: '2022',
                value: it['Производство электроэнергии на душу (кВт·ч/чел) 2022'],
              },
            ],
          }));

        this.lineData1 = data1.filter(it => it.name.includes('федеральный округ'))
          .map(it => ({...it, name: it.name.replace(' федеральный округ', '')}));
        this.lineData2 = data2.filter(it => it.name.includes('федеральный округ'))
          .map(it => ({...it, name: it.name.replace(' федеральный округ', '')}));
        this.lineData3 = data3.filter(it => it.name.includes('федеральный округ'))
          .map(it => ({...it, name: it.name.replace(' федеральный округ', '')}));
        this.lineData4 = data4.filter(it => it.name.includes('федеральный округ'))
          .map(it => ({...it, name: it.name.replace(' федеральный округ', '')}));
        this.lineData5 = data5.filter(it => it.name.includes('федеральный округ'))
          .map(it => ({...it, name: it.name.replace(' федеральный округ', '')}));
        this.lineData6 = data6.filter(it => it.name.includes('федеральный округ'))
          .map(it => ({...it, name: it.name.replace(' федеральный округ', '')}));
      },
      error: (error) => {
        console.error('Error in component subscription:', error);
      }
    });
  }

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

  protected readonly LegendPosition = LegendPosition;
}
