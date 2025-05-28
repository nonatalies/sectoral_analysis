import {Component, Inject} from '@angular/core';
import {MAT_DIALOG_DATA} from '@angular/material/dialog';

@Component({
  selector: 'app-table-dialog',
  standalone: false,
  templateUrl: './table-dialog.component.html',
  styleUrl: './table-dialog.component.scss'
})
export class TableDialogComponent {
  constructor(@Inject(MAT_DIALOG_DATA) public data: any[]) {}

}
