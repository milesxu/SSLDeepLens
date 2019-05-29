import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

export class ImageNumber {
  baseName: string;
  start: number;
  length: number;
  mask: number[];
}

@Injectable({
  providedIn: 'root'
})
export class LoadService {
  private ImageNumberSource = new BehaviorSubject<ImageNumber>({
    baseName: 'assets/combine213/',
    start: 110000,
    length: 128,
    mask: new Array(128).fill(1)
  });
  imageNumber$ = this.ImageNumberSource.asObservable();

  reloadImage(imageNum: number) {
    const name = 'assets/combine213/';
    const begin = 110000;
    const offset = Math.floor(Math.random() * (10000 - imageNum));
    // console.log(offset);
    this.ImageNumberSource.next({
      baseName: name,
      start: begin + offset,
      length: imageNum,
      mask: new Array(imageNum).fill(1)
    });
  }

  reloadImageAfterRun(imgNum: ImageNumber) {
    this.ImageNumberSource.next(imgNum);
  }

  constructor() {}

  get imageNumber() {
    return this.ImageNumberSource.getValue();
  }
}
