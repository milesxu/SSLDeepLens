import { Component } from '@angular/core';
import { Record } from '../record';
import { RecordService } from '../record.service';

@Component({
  selector: 'app-records',
  templateUrl: './records.component.html'
})
export class RecordsComponent {
  sortName: string | null = null;
  sortValue: string | null = null;
  searchAddress: string;
  listOfName = [{ text: 'Joe', value: 'Joe' }, { text: 'Jim', value: 'Jim' }];
  listOfAddress = [
    { text: 'London', value: 'London' },
    { text: 'Sidney', value: 'Sidney' }
  ];
  // listOfDataset
  listOfProcessor = [
    { text: 'CPU', value: 'CPU' },
    { text: 'GPU', value: 'GPU' }
  ];
  listOfSearchName: string[] = [];
  listOfData: Array<{
    id: number;
    dataset: string;
    processor: string;
    algorithm: string;
    images: number;
    timing: number;
    speed: number;
    accuracy: number;
    [key: string]: string | number;
  }> = [];
  /*{
      id: 1,
      dataset: 'Sky based',
      processor: 'CPU',
      algorithm: 'Resnet50',
      images: 1024,
      timing: 40.0,
      speed: 2.0,
      accuracy: 0.88
    },
    {
      id: 2,
      dataset: 'Ground based',
      processor: 'GPU',
      algorithm: 'Resnet50 + SNTG',
      images: 1024,
      timing: 10.0,
      speed: 20.0,
      accuracy: 0.92
    },
    {
      id: 3,
      dataset: 'Sky based',
      processor: 'CPU',
      algorithm: 'Resnet50',
      images: 1024,
      timing: 40.0,
      speed: 2.0,
      accuracy: 0.88
    },
    {
      id: 4,
      dataset: 'Sky based',
      processor: 'CPU',
      algorithm: 'Resnet50',
      images: 1024,
      timing: 40.0,
      speed: 2.0,
      accuracy: 0.88
    }
  ];*/
  listOfDisplayData: Array<{
    id: number;
    dataset: string;
    processor: string;
    algorithm: string;
    images: number;
    timing: number;
    speed: number;
    accuracy: number;
    [key: string]: string | number;
  }> = [...this.listOfData];

  /*sort(sort: { key: string; value: string }): void {
    this.sortName = sort.key;
    this.sortValue = sort.value;
    this.search();
  }

  filter(listOfSearchName: string[], searchAddress: string): void {
    this.listOfSearchName = listOfSearchName;
    this.searchAddress = searchAddress;
    this.search();
  }

  search(): void {
    // filter data
    const filterFunc = (item: { name: string; age: number; address: string }) =>
      (this.searchAddress
        ? item.address.indexOf(this.searchAddress) !== -1
        : true) &&
      (this.listOfSearchName.length
        ? this.listOfSearchName.some(name => item.name.indexOf(name) !== -1)
        : true);
    const data = this.listOfData.filter(item => filterFunc(item));
    //  sort data
    if (this.sortName && this.sortValue) {
      this.listOfDisplayData = data.sort((a, b) =>
        this.sortValue === 'ascend'
          ? a[this.sortName!] > b[this.sortName!]
            ? 1
            : -1
          : b[this.sortName!] > a[this.sortName!]
          ? 1
          : -1
      );
    } else {
      this.listOfDisplayData = data;
    }
  }*/

  constructor(private recordService: RecordService) {
    recordService.recordObservable.subscribe(record => {
      console.log(record);
      const id = this.listOfData.length + 1;
      this.listOfData.push({
        id: id,
        dataset: record.dataset,
        processor: record.processor,
        algorithm: record.algorithm,
        images: record.imageSize,
        timing: record.timing,
        speed: record.speed,
        accuracy: record.accuracy
      });
      this.listOfDisplayData = [...this.listOfData];
    });
  }
}
