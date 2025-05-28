import { Injectable } from '@angular/core';
import {HttpClient} from '@angular/common/http';
import * as XLSX from 'xlsx';
import {BehaviorSubject, Observable} from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class AppService {
  private tableDataSubject = new BehaviorSubject<any[]>([]);

  public tableData$: Observable<any[]> = this.tableDataSubject.asObservable();

  private statDataSubject = new BehaviorSubject<any[]>([]);

  public statData$: Observable<any[]> = this.statDataSubject.asObservable();

  private districtDataSubject = new BehaviorSubject<any[]>([]);

  public districtData$: Observable<any[]> = this.districtDataSubject.asObservable();

  private trendDataSubject = new BehaviorSubject<any[]>([]);

  public trendData$: Observable<any[]> = this.trendDataSubject.asObservable();

  constructor(private http: HttpClient) {
    this.loadTableData();
  }

  public loadTableData(): void {
    const url = 'data/table_info.xlsx';

    this.http.get(url, { responseType: 'arraybuffer' }).subscribe((arrayBuffer: ArrayBuffer) => {
      const data = new Uint8Array(arrayBuffer);
      const workbook = XLSX.read(data, { type: 'array' });

      this.tableDataSubject.next(XLSX.utils.sheet_to_json(workbook.Sheets['table']));
      this.districtDataSubject.next(XLSX.utils.sheet_to_json(workbook.Sheets['district']));
      this.statDataSubject.next(XLSX.utils.sheet_to_json(workbook.Sheets['stat']));
      this.trendDataSubject.next(XLSX.utils.sheet_to_json(workbook.Sheets['trend']));
    });
  }
}
