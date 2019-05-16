import { Injectable } from '@angular/core';
import { Subject } from 'rxjs';
import { Record } from './record';

@Injectable({
  providedIn: 'root'
})
export class RecordService {
  private recordSource = new Subject<Record>();

  recordObservable = this.recordSource.asObservable();

  addRecord(record: Record): void {
    // console.log(record);
    this.recordSource.next(record);
  }
  constructor() {}
}
