import {AfterViewInit, Component, inject, ViewChild} from '@angular/core';
import {AppService} from '../app.service';
import {MatTableDataSource} from '@angular/material/table';
import {MatSort} from '@angular/material/sort';
import {MatDialog} from '@angular/material/dialog';
import {TableDialogComponent} from './table-dialog/table-dialog.component';

@Component({
  selector: 'app-table',
  standalone: false,
  templateUrl: './table.component.html',
  styleUrl: './table.component.scss'
})
export class TableComponent implements AfterViewInit {
  public displayedColumns: string[] = ['name', 'type', 'number', 'value1', 'value2', 'value3', 'value4', 'value5', 'value6'];
  public dataSource = new MatTableDataSource<any>();
  public dataForDialog: any[] = [];
  @ViewChild(MatSort) sort!: MatSort;

  ngAfterViewInit() {
    this.dataSort();
  }

  constructor(private appService: AppService) {
    this.appService.tableData$.subscribe({
      next: (data) => {
        this.dataSource.data = data.map((item: any) => {
          const cleanedItem: any = {};
          for (const key in item) {
            const value = item[key];
            cleanedItem[key] = value === 'nan' ? null : value;
          }
          return cleanedItem;
        });
        this.dataSort();
        this.dataForDialog = [{
          name: 'Существующий',
          series: data.filter(it => it['Тип'] === 'Существующий').map(item => ({
            name: item['Регион'],
            value: item['Комплексный показатель']
          })),
        },
          {
            name: 'Новый',
            series: data.filter(it => it['Тип'] === 'Новый').map(item => ({
              name: item['Регион'],
              value: item['Комплексный показатель']
            })),
          },
        ]
      },
      error: (error) => {
        console.error('Error in component subscription:', error);
      }
    });
  }

  dataSort() {
    this.dataSource.sort = this.sort;
    this.dataSource.sortingDataAccessor = (item, property) => {
      switch (property) {
        case 'name':
          return item['Регион'];
        case 'type':
          return item['Тип'];
        case 'number':
          return item['Комплексный показатель'];
        case 'value1':
          return item['Посевные площади (тыс. га)'];
        case 'value2':
          return item['Индекс сельского хозяйства'];
        case 'value3':
          return item['Добыча полезных ископаемых (млн руб)'];
        case 'value4':
          return item['Индекс пром. производства'];
        case 'value5':
          return item['Потребление электроэнергии (млн кВт·ч)'];
        case 'value6':
          return item['Производство электроэнергии на душу (кВт·ч/чел)'];
        default:
          return item[property];
      }
    };
  }

  applyFilter(event: Event) {
    const filterValue = (event.target as HTMLInputElement).value;
    this.dataSource.filter = filterValue.trim().toLowerCase();
  }

  readonly dialog = inject(MatDialog);

  openDialog() {
    this.dialog.open(TableDialogComponent, {
      data: this.dataForDialog
    });
  }
}
